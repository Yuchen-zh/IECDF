import os
import tqdm
import torch
from transformers import BertModel
import torch.nn as nn
import models_mae
import pandas as pd

from utils.utils import data2gpu, Averager, metrics, Recorder, clipdata2gpu
from utils.utils import metricsTrueFalse, extract_and_visualize
from .layers import *
from .Mamba_Family import *
from .SelfAttention_Family import *
from .loss import *
from mamba_ssm import Mamba
from timm.models.vision_transformer import Block
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

class IECDF(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout, category_dict):
        super(IECDF, self).__init__()
        self.num_expert = 3
        self.domain_num = 9
        self.num_share = 3
        self.unified_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.text_token_len = 197
        self.image_token_len = 197
        self.category_dict = category_dict

        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))

            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
        self.text_experts = nn.ModuleList(text_expert_list)

        image_expert_list = []
        for i in range(self.domain_num):
            image_expert = []
            for j in range(self.num_expert):
                image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)
        self.image_experts = nn.ModuleList(image_expert_list)

        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 320),
                                       )
                fusion_expert.append(expert)
            fusion_expert = nn.ModuleList(fusion_expert)
            fusion_expert_list.append(fusion_expert)
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        final_expert_list = []
        for i in range(self.domain_num):
            final_expert = []
            for j in range(self.num_expert):
                final_expert.append(Block(dim=320, num_heads=8))
            final_expert = nn.ModuleList(final_expert)
            final_expert_list.append(final_expert)
        self.final_experts = nn.ModuleList(final_expert_list)

        text_share_expert, image_share_expert, fusion_share_expert, final_share_expert = [], [], [], []
        for i in range(self.num_share):
            text_share = []
            image_share = []
            fusion_share = []
            final_share = []
            for j in range(self.num_expert):
                text_share.append(cnn_extractor(emb_dim, feature_kernel))
                image_share.append(cnn_extractor(self.image_dim, feature_kernel))
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 320),
                                       )
                fusion_share.append(expert)
                final_share.append(Block(dim=320, num_heads=8))
            text_share = nn.ModuleList(text_share)
            text_share_expert.append(text_share)
            image_share = nn.ModuleList(image_share)
            image_share_expert.append(image_share)
            fusion_share = nn.ModuleList(fusion_share)
            fusion_share_expert.append(fusion_share)
            final_share = nn.ModuleList(final_share)
            final_share_expert.append(final_share)
        self.text_share_expert = nn.ModuleList(text_share_expert)
        self.image_share_expert = nn.ModuleList(image_share_expert)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
        self.final_share_expert = nn.ModuleList(final_share_expert)

        image_gate_list, text_gate_list, share_gate_list1, share_gate_list2, fusion_gate_list, final_gate_list = [], [], [], [], [], []
        for i in range(self.domain_num):
            image_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.unified_dim, self.num_expert),
                                       nn.Dropout(0.1),
                                       nn.Softmax(dim=1)
                                       )
            text_gate = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.unified_dim, self.num_expert),
                                      nn.Dropout(0.1),
                                      nn.Softmax(dim=1)
                                      )
            fusion_gate = nn.Sequential(nn.Linear(320, 160),
                                         nn.SiLU(),
                                         nn.Linear(160, self.num_expert),
                                         nn.Dropout(0.1),
                                         nn.Softmax(dim=1)
                                         )
            final_gate = nn.Sequential(nn.Linear(1088, 720),
                                        nn.SiLU(),
                                        nn.Linear(720, 160),
                                        nn.SiLU(),
                                        nn.Linear(160, self.num_expert * 3),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                         )
            image_gate_list.append(image_gate)
            text_gate_list.append(text_gate)
            fusion_gate_list.append(fusion_gate)
            final_gate_list.append(final_gate)
        for i in range(self.domain_num):
            share_gate1 = nn.Sequential(nn.Linear(self.unified_dim * 2, self.unified_dim),
                                        nn.SiLU(),
                                        nn.Linear(self.unified_dim, self.num_expert),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                        )
            share_gate2 = nn.Sequential(nn.Linear(320, 160),
                                        nn.SiLU(),
                                        nn.Linear(160, self.num_expert),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                        )
            share_gate_list1.append(share_gate1)
            share_gate_list2.append(share_gate2)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.fusion_gate_list = nn.ModuleList(fusion_gate_list)
        self.share_gate_list1 = nn.ModuleList(share_gate_list1)
        self.share_gate_list2 = nn.ModuleList(share_gate_list2)
        self.final_gate_list = nn.ModuleList(final_gate_list)

        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.fusion_attention = TokenAttention(self.unified_dim * 2)

        self.domain_embedder = torch.nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)

        text_classifier_list = []
        for i in range(self.domain_num):
            text_classifier = MLP(320, mlp_dims, dropout)
            text_classifier_list.append(text_classifier)
        self.text_classifier_list = nn.ModuleList(text_classifier_list)

        image_classifier_list = []
        for i in range(self.domain_num):
            image_classifier = MLP(320, mlp_dims, dropout)
            image_classifier_list.append(image_classifier)
        self.image_classifier_list = nn.ModuleList(image_classifier_list)

        fusion_classifier_list = []
        for i in range(self.domain_num):
            fusion_classifier = MLP(320, mlp_dims, dropout)
            fusion_classifier_list.append(fusion_classifier)
        self.fusion_classifier_list = nn.ModuleList(fusion_classifier_list)

        final_classifier_list = []
        for i in range(self.domain_num):
            final_classifier = MLP(1920, mlp_dims, dropout)
            final_classifier_list.append(final_classifier)
        self.final_classifier_list = nn.ModuleList(final_classifier_list)

        self.gate_expert = nn.ModuleList(
            [MLP_gate_fusion(197*768, 320, 
                        [1024], 0.1) for i in range(self.domain_num+1)])
        self.gate_fusion_expert = nn.ModuleList(
            [MLP_share_gate_fusion(320, 320, [768], 0.1) for _ in range(self.domain_num)])

        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(768 * 2, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(1088, 320, [348], 0.1)
        self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)

        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        self.image_model.cuda()
        checkpoint = torch.load('/ssd/home/project/IECDF/mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False

        self.ClipModel,_ = load_from_name("ViT-B-16", device="cuda", download_root='./')

        feature_emb_size = 320
        img_emb_size =320
        feature_num = 6
        self.feature_num = 6
        text_emb_size = 320
        self.feature_emb_size = 320
        self.emb_size = 320
        self.layers = 12
        self.mlp_img = torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
                                                 range(feature_num)])

        self.mlp_text = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                            range(feature_num)])
        self.mlp_fusion = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
                                             range(feature_num)])
        
        mamba_list = []
        for i in range(1):
            mamba_list.append(AM_Layer(
                    AttentionLayer(
                        FullAttention(True, factor=1, attention_dropout=0.05, output_attention=False),
                        d_model=320, n_heads=8),
                    Mamba(d_model=320, d_state=6, d_conv=4, expand=2),
                    d_model=320,
                    dropout=0.05
                ))
            
        self.AM_layers = torch.nn.ModuleList(mamba_list)

        self.fusion_conv = nn.Conv1d(
                in_channels=640,
                out_channels=320,
                kernel_size=1
            )

        self.gate_att = nn.Linear(640, 320)

    def to_sequence(self, text_emb, image_emb, fusion_emb, mlp_img, mlp_text, mlp_fusion):
        for img_feature_num in range(0, self.feature_num):
            if img_feature_num == 0:
                img_feature_seq = mlp_img[img_feature_num](image_emb)
                img_feature_seq = img_feature_seq.unsqueeze(1)
            else:
                img_feature_seq = torch.cat((img_feature_seq, mlp_img[img_feature_num](image_emb).unsqueeze(1)), 1)

        for text_feature_num in range(0, self.feature_num):
            if text_feature_num == 0:
                text_feature_seq = mlp_text[text_feature_num](text_emb)
                text_feature_seq = text_feature_seq.unsqueeze(1)
            else:
                text_feature_seq = torch.cat((text_feature_seq, mlp_text[text_feature_num](text_emb).unsqueeze(1)), 1)

        for text_feature_num in range(0, self.feature_num):
            if text_feature_num == 0:
                fusion_feature_seq = mlp_fusion[text_feature_num](fusion_emb)
                fusion_feature_seq = fusion_feature_seq.unsqueeze(1)
            else:
                fusion_feature_seq = torch.cat((fusion_feature_seq, mlp_fusion[text_feature_num](fusion_emb).unsqueeze(1)), 1)

        combined_tmp_value = (text_feature_seq, img_feature_seq, fusion_feature_seq)

        return combined_tmp_value 

    def mamba(self,x1, x2, x3):
        for AM_layers_list in self.AM_layers:
            clip_temp_feature = AM_layers_list(x1)
            text_temp_feature = AM_layers_list(x2, clip_temp_feature)
            image_temp_feature = AM_layers_list(x3, clip_temp_feature)
        
        return text_temp_feature, image_temp_feature



    def forward(self, **kwargs):
        inputs = kwargs['content'] #[64,197]
        masks = kwargs['content_masks'] #[64,197]
        category = kwargs['category'] #[64]
        text_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])
        image = kwargs['image'] # [64, 3, 224, 224]
        image_feature = self.image_model.forward_ying(image)  # ([64, 197, 768])
        clip_image = kwargs['clip_image'] # [64, 3, 224, 224]
        clip_text = kwargs['clip_text'] # [64,52]
        
        with torch.no_grad():
            clip_image_feature = self.ClipModel.encode_image(clip_image)# ([64, 512])
            clip_text_feature = self.ClipModel.encode_text(clip_text)  # ([64, 512])
            clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True) # ([64, 512])
            clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True) # ([64, 512])
        clip_fusion_feature = torch.cat((clip_image_feature, clip_text_feature),dim=-1)#torch.Size([64, 1024])
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())#torch.Size([64, 320])

        text_atn_feature = self.text_attention(text_feature,masks) # ([64, 768])
        image_atn_feature, _ = self.image_attention(image_feature) # ([64, 768])

        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda() # ([64, 1])
        domain_embedding = self.domain_embedder(idxs).squeeze(1)  ##([64, 768])
        text_gate_input = torch.cat([domain_embedding, text_atn_feature], dim=-1)  # ([64, 1536])
        image_gate_input = torch.cat([domain_embedding, image_atn_feature], dim=-1) # ([64, 1536])

        text_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list[i](text_gate_input)
            text_gate_out_list.append(gate_out)
        self.text_gate_out_list = text_gate_out_list

        text_share_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.share_gate_list1[i](text_gate_input)
            text_share_gate_out_list.append(gate_out)
        self.text_share_gate_out_list = text_share_gate_out_list

        id = torch.tensor([index for index in category]).cuda()
        text_category_features = [torch.zeros(0, 197, 768, device='cuda') for _ in range(9)] 

        for i in range(self.domain_num):
            idx = (id == i).nonzero(as_tuple=True)[0]

            if idx.numel() > 0:
                text_category_features[i] = text_feature[idx]
            else:
                text_category_features[i] = torch.zeros(1, 197, 768, device='cuda')

        text_special_expert_outputs = [] 
        for i in range(self.domain_num):
            text_special_expert_value = []
            for j in range(self.num_expert):
                expert_output = self.text_experts[i][j](text_category_features[i])
                padding = (0, 0, 0, 64 - expert_output.size(0))
                expert_output_padded = F.pad(expert_output, padding, "constant", 0)
                text_special_expert_value.append(expert_output_padded)
                
            text_special_expert_outputs.append(text_special_expert_value)

        text_shared_expert_outputs = []
        for i in range(self.num_expert):
            text_shared_expert_outputs.append(self.text_share_expert[0][i](text_feature))

        text_gate_expert_stem_value = []
        text_gate_special_expert_stem_value = []
        text_gate_share_expert_stem_value = []
        for i in range(self.domain_num):
            gate_input = []
            for j in range(self.domain_num):
                if j == i:
                    gate_input.append(text_special_expert_outputs[j])
                else: 
                    specific_expert_outputs_j = text_special_expert_outputs[j]
                    specific_expert_outputs_j = [out.detach() for out in specific_expert_outputs_j]
                    gate_input.append(specific_expert_outputs_j)
                    del specific_expert_outputs_j

            gate_input.append(text_shared_expert_outputs)

            gate_expert = 0
            gate_special_expert = 0
            gate_share_expert = 0


            for n in range(self.num_expert):
                gate_expert += (text_gate_out_list[i][:, n].unsqueeze(1) * gate_input[i][n])
                gate_special_expert += (text_gate_out_list[i][:, n].unsqueeze(1) * gate_input[i][n])
            for p in range(self.num_expert):
                gate_expert += (text_share_gate_out_list[i][:, p].unsqueeze(1) * gate_input[self.domain_num][p])
                gate_share_expert += (text_share_gate_out_list[i][:, p].unsqueeze(1) * gate_input[self.domain_num][p])
            text_gate_expert_stem_value.append(gate_expert)
            text_gate_special_expert_stem_value.append(gate_special_expert)
            text_gate_share_expert_stem_value.append(gate_share_expert)

            torch.cuda.empty_cache()

        text_label_pred = []
        for i in range(self.domain_num):
            text_class = self.text_classifier_list[i](text_gate_expert_stem_value[i]).squeeze(1)
            probs = torch.softmax(text_class, dim=1)
            text_label_pred.append(probs)
        text_label_pred_list = []
        text_label_pred_avg = 0
        for i in range(self.domain_num):
            text_label_pred_list.append(text_label_pred[i][idxs.squeeze() == i])
            text_label_pred_avg += text_label_pred[i]
        text_label_pred_avg = text_label_pred_avg / 9
        text_label_pred_list = torch.cat((text_label_pred_list[0], text_label_pred_list[1], text_label_pred_list[2], text_label_pred_list[3],
                                     text_label_pred_list[4], text_label_pred_list[5], text_label_pred_list[6], text_label_pred_list[7], text_label_pred_list[8]))
        

        image_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.image_gate_list[i](image_gate_input) # ([64, 18])
            image_gate_out_list.append(gate_out) # len:9
        self.image_gate_out_list = image_gate_out_list

        image_share_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.share_gate_list1[i](image_gate_input)
            image_share_gate_out_list.append(gate_out)
        self.image_share_gate_out_list = image_share_gate_out_list

        id = torch.tensor([index for index in category]).cuda()
        image_category_features = [torch.zeros(0, 197, 768, device='cuda') for _ in range(9)]

        for i in range(self.domain_num):
            idx = (id == i).nonzero(as_tuple=True)[0]

            if idx.numel() > 0:
                image_category_features[i] = image_feature[idx]
            else:
                image_category_features[i] = torch.zeros(1, 197, 768, device='cuda')

        image_special_expert_outputs = [] 
        for i in range(self.domain_num):
            image_special_expert_value = []
            for j in range(self.num_expert):
                expert_output = self.image_experts[i][j](image_category_features[i])
                padding = (0, 0, 0, 64 - expert_output.size(0))
                expert_output_padded = F.pad(expert_output, padding, "constant", 0)
                image_special_expert_value.append(expert_output_padded)
                
            image_special_expert_outputs.append(image_special_expert_value)

        image_shared_expert_outputs = []
        for i in range(self.num_expert):
            image_shared_expert_outputs.append(self.image_share_expert[0][i](image_feature))

        image_gate_expert_stem_value = []
        image_gate_special_expert_stem_value = []
        image_gate_share_expert_stem_value = []        
        for i in range(self.domain_num):
            gate_input = []
            for j in range(self.domain_num):
                if j == i:
                    gate_input.append(image_special_expert_outputs[j])
                else: 
                    specific_expert_outputs_j = image_special_expert_outputs[j]
                    specific_expert_outputs_j = [out.detach() for out in specific_expert_outputs_j]
                    gate_input.append(specific_expert_outputs_j)
                    del specific_expert_outputs_j

            gate_input.append(image_shared_expert_outputs)
            
            gate_expert = 0
            gate_special_expert = 0
            gate_share_expert = 0


            for n in range(self.num_expert):
                gate_expert += (image_gate_out_list[i][:, n].unsqueeze(1) * gate_input[i][n])
                gate_special_expert += (image_gate_out_list[i][:, n].unsqueeze(1) * gate_input[i][n])
            for p in range(self.num_expert):
                gate_expert += (image_share_gate_out_list[i][:, p].unsqueeze(1) * gate_input[self.domain_num][p])
                gate_share_expert += (image_share_gate_out_list[i][:, p].unsqueeze(1) * gate_input[self.domain_num][p])
            image_gate_expert_stem_value.append(gate_expert)
            image_gate_special_expert_stem_value.append(gate_special_expert)
            image_gate_share_expert_stem_value.append(gate_share_expert)

            torch.cuda.empty_cache()


        image_label_pred = []
        for i in range(self.domain_num):
            image_class = self.image_classifier_list[i](image_gate_expert_stem_value[i]).squeeze(1)
            probs = torch.softmax(image_class, dim=1)
            image_label_pred.append(probs)
        image_label_pred_list = []
        image_label_pred_avg = 0
        for i in range(self.domain_num):
            image_label_pred_list.append(image_label_pred[i][idxs.squeeze() == i])
            image_label_pred_avg += image_label_pred[i]
        image_label_pred_avg = image_label_pred_avg / 9
        image_label_pred_list = torch.cat((image_label_pred_list[0], image_label_pred_list[1], image_label_pred_list[2], image_label_pred_list[3],
                                    image_label_pred_list[4], image_label_pred_list[5], image_label_pred_list[6], image_label_pred_list[7], image_label_pred_list[8]))


        text = text_gate_share_expert_stem_value[0]
        image = image_gate_share_expert_stem_value[0]

        fusion_share_feature = torch.cat((clip_fusion_feature, text, image), dim=-1)
        fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        fusion_gate_input = self.domain_fusion(torch.cat([domain_embedding, fusion_share_feature], dim=-1))
        fusion_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.fusion_gate_list[i](fusion_gate_input)
            fusion_gate_out_list.append(gate_out)
        self.fusion_gate_out_list = fusion_gate_out_list

        fusion_share_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.share_gate_list2[i](fusion_gate_input)
            fusion_share_gate_out_list.append(gate_out)
        self.fusion_share_gate_out_list = fusion_share_gate_out_list

        id = torch.tensor([index for index in category]).cuda()
        fusion_category_features = [torch.zeros(0, 320, device='cuda') for _ in range(9)]

        for i in range(self.domain_num):
            idx = (id == i).nonzero(as_tuple=True)[0]

            if idx.numel() > 0:
                fusion_category_features[i] = fusion_share_feature[idx]
            else:
                fusion_category_features[i] = torch.zeros(1, 320, device='cuda')

        fusion_special_expert_outputs = [] 
        for i in range(self.domain_num):
            fusion_special_expert_value = []
            for j in range(self.num_expert):
                expert_output = self.fusion_experts[i][j](fusion_category_features[i])
                padding = (0, 0, 0, 64 - expert_output.size(0))
                expert_output_padded = F.pad(expert_output, padding, "constant", 0)
                fusion_special_expert_value.append(expert_output_padded)
                
            fusion_special_expert_outputs.append(fusion_special_expert_value)

        fusion_shared_expert_outputs = []
        for i in range(self.num_expert):
            fusion_shared_expert_outputs.append(self.fusion_share_expert[0][i](fusion_share_feature))

        
        fusion_gate_expert_stem_value = []
        fusion_gate_special_expert_stem_value = []
        fusion_gate_share_expert_stem_value = []
        for i in range(self.domain_num):
            gate_input = []
            for j in range(self.domain_num):
                if j == i:
                    gate_input.append(fusion_special_expert_outputs[j])
                else: 
                    specific_expert_outputs_j = fusion_special_expert_outputs[j]
                    specific_expert_outputs_j = [out.detach() for out in specific_expert_outputs_j]
                    gate_input.append(specific_expert_outputs_j)
                    del specific_expert_outputs_j
            
            gate_input.append(fusion_shared_expert_outputs)

            gate_expert = 0
            gate_special_expert = 0
            gate_share_expert = 0


            for n in range(self.num_expert):
                gate_expert += (fusion_gate_out_list[i][:, n].unsqueeze(1) * gate_input[i][n])
                gate_special_expert += (fusion_gate_out_list[i][:, n].unsqueeze(1) * gate_input[i][n])
            for p in range(self.num_expert):
                gate_expert += (fusion_share_gate_out_list[i][:, p].unsqueeze(1) * gate_input[self.domain_num][p])
                gate_share_expert += (fusion_share_gate_out_list[i][:, p].unsqueeze(1) * gate_input[self.domain_num][p])
            fusion_gate_expert_stem_value.append(gate_expert)
            fusion_gate_special_expert_stem_value.append(gate_special_expert)
            fusion_gate_share_expert_stem_value.append(gate_share_expert)
        
            torch.cuda.empty_cache()
       

        fusion_label_pred = []
        for i in range(self.domain_num):
            fusion_class = self.fusion_classifier_list[i](fusion_gate_expert_stem_value[i]).squeeze(1)
            probs = torch.softmax(fusion_class, dim=1)
            fusion_label_pred.append(probs)
        fusion_label_pred_list = []
        fusion_label_pred_avg = 0
        for i in range(self.domain_num):
            fusion_label_pred_list.append(fusion_label_pred[i][idxs.squeeze() == i])
            fusion_label_pred_avg += fusion_label_pred[i]
        fusion_label_pred_avg = fusion_label_pred_avg / 9
        fusion_label_pred_list = torch.cat((fusion_label_pred_list[0], fusion_label_pred_list[1], fusion_label_pred_list[2], fusion_label_pred_list[3],
                                    fusion_label_pred_list[4], fusion_label_pred_list[5], fusion_label_pred_list[6], fusion_label_pred_list[7], fusion_label_pred_list[8]))

        text_gate_share_value = text_gate_share_expert_stem_value[0]
        image_gate_share_value = image_gate_share_expert_stem_value[0]
        fusion_gate_share_value = fusion_gate_share_expert_stem_value[0]

        combined_share_value = self.to_sequence(text_gate_share_value, image_gate_share_value, fusion_gate_share_value, self.mlp_img, self.mlp_text, self.mlp_fusion)

        combined_specific_value = []
        for i in range(self.domain_num):
            combined_tmp_value = self.to_sequence(text_gate_special_expert_stem_value[i], 
                                  image_gate_special_expert_stem_value[i], 
                                  fusion_gate_special_expert_stem_value[i],
                                  self.mlp_img, self.mlp_text, self.mlp_fusion)
            combined_specific_value.append(combined_tmp_value)
        combined_specific_value.append(combined_share_value)

        fusion_output = []
        for i in range(self.domain_num):
            for j in range(3):
                if j == 0:
                    text_feature, image_feature = self.mamba(combined_specific_value[i][2], combined_specific_value[i][0], combined_specific_value[i][1])
                    temp_feature = torch.cat((text_feature, image_feature), dim=2)
                    f_att = torch.sigmoid(self.gate_att(temp_feature))
                    temp_feature = f_att * text_feature + (1 - f_att) * image_feature
                else:
                    text_feature, image_feature = self.mamba(temp_feature, combined_specific_value[i][0], combined_specific_value[i][1])
                    temp_feature = torch.cat((text_feature, image_feature), dim=2)
                    f_att = torch.sigmoid(self.gate_att(temp_feature))
                    temp_feature = f_att * text_feature + (1 - f_att) * image_feature
            
            fusion_output.append(temp_feature)

        fusion_share_output = []
        for j in range(3):
            if j == 0:
                text_feature, image_feature = self.mamba(combined_share_value[2], combined_share_value[0], combined_share_value[1])
                temp_feature = torch.cat((text_feature, image_feature), dim=2)
                f_att = torch.sigmoid(self.gate_att(temp_feature))
                temp_feature = f_att * text_feature + (1 - f_att) * image_feature
            else:
                text_feature, image_feature = self.mamba(temp_feature, combined_share_value[0], combined_share_value[1])
                temp_feature = torch.cat((text_feature, image_feature), dim=2)
                f_att = torch.sigmoid(self.gate_att(temp_feature))
                temp_feature = f_att * text_feature + (1 - f_att) * image_feature
        
        fusion_share_output.append(temp_feature)

        fusion_final_value = []
        fusion_final_features = []
        fusion_final_labels = []
        for i in range(self.domain_num):
            fusion_share_temp_feature = torch.cat((fusion_output[i], fusion_share_output[0]), dim=2)
            fusion_share_temp_feature = fusion_share_temp_feature.permute(0, 2, 1)
            output = self.fusion_conv(fusion_share_temp_feature)
            output = output.view(64, -1)
            logits = self.final_classifier_list[i](output)
            probs = torch.softmax(logits, dim=1)
            fusion_final_value.append(probs)
            fusion_final_features.append(output.detach())
            fusion_final_labels.append(torch.argmax(torch.softmax(logits, dim=1), dim=1))
        final_label_pred = []
        final_label_label = []
        for i in range(self.domain_num):
            final_label_pred.append(fusion_final_features[i][idxs.squeeze() == i])
            final_label_label.append(fusion_final_labels[i][idxs.squeeze() == i])
        
        final_label_pred_list = []
        final_label_pred_avg = 0
        for i in range(self.domain_num):
            final_label_pred_list.append(fusion_final_value[i][idxs.squeeze() == i])
            final_label_pred_avg += fusion_final_value[i]
        final_label_pred_avg = final_label_pred_avg / 9
        final_label_pred_list = torch.cat((final_label_pred_list[0], final_label_pred_list[1], final_label_pred_list[2], final_label_pred_list[3],
                                     final_label_pred_list[4], final_label_pred_list[5], final_label_pred_list[6], final_label_pred_list[7],final_label_pred_list[8]))

        return final_label_pred_list, fusion_label_pred_list, image_label_pred_list, text_label_pred_list

class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 batchsize,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 category_label_distribution,
                 early_stop=5,
                 epoches=100
                 ):
        self.batch_size = batchsize
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        self.category_label_distribution = category_label_distribution
        if not os.path.exists(save_param_dir):
            self.save_param_dir = os.makedirs(save_param_dir)
        else:
            self.save_param_dir = save_param_dir

    def train(self):
        self.model = IECDF(self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout, self.category_dict)
        if self.use_cuda:
            self.model = self.model.cuda()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop)

        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            avg_minority_loss = Averager()
            avg_majority_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = clipdata2gpu(batch)
                batch_size_actual = len(batch_data['content'])

                if batch_size_actual < self.train_loader.batch_size:
                    batch_needed = self.train_loader.batch_size - batch_size_actual
                    print(f"not enough, add {batch_needed} data")

                    remaining_data_iter = iter(self.train_loader)

                    additional_data = next(remaining_data_iter)
                    
                    additional_data_batch = {
                        key: value[:batch_needed].cuda() for key, value in zip(
                            ['content', 'content_masks', 'label', 'category', 'image', 'clip_image', 'clip_text'],
                            additional_data
                        )
                    }

                    for key in batch_data:
                        batch_data[key] = torch.cat([batch_data[key]] + [additional_data_batch[key]], dim=0)

                    print(f"add {batch_needed} data")

                class_counts = [[0, 0] for _ in range(9)]
                label = batch_data['label']
                category = batch_data['category']
                for i in range(9):
                    class_counts[i][0] += (label[(category == i)] == 0).sum().item()
                    class_counts[i][1] += (label[(category == i)] == 1).sum().item()

                idxs = torch.tensor([index for index in category]).view(-1, 1)
                batch_label = torch.cat([
                                    label[idxs.squeeze() == i] 
                                    for i in range(9)
                                ])
                final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list = self.model(**batch_data)
                batch_size = batch_label.size(0)
                batch_label_onehot = torch.zeros(batch_size, 2, device=batch_label.device)
                batch_label_onehot[range(batch_size), batch_label.long()] = 1
                loss_fn = DLINEXLoss(
                                a=2.0,
                                category_label_distribution=self.category_label_distribution,
                                categorys=9,
                                n_classes=2,
                                device=batch_label.device
                        )
                loss0, minority0, majority0 = loss_fn(final_label_pred_list, batch_label_onehot, category)
                loss1, minority1, majority1 = loss_fn(fusion_label_pred_list, batch_label_onehot, category)
                loss2, minority2, majority2 = loss_fn(image_label_pred_list, batch_label_onehot, category)
                loss3, minority3, majority3 = loss_fn(text_label_pred_list, batch_label_onehot, category)
                loss = 0.7*loss0+0.1*loss1+0.1*loss2+0.1*loss3
                loss_minority = 0.7*minority0+0.1*minority1+0.1*minority2+0.1*minority3
                loss_majority = 0.7*majority0+0.1*majority1+0.1*majority2+0.1*majority3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
                avg_minority_loss.add(loss_minority.item())
                avg_majority_loss.add(loss_majority.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            print('Training Epoch {}; Minority_Loss {}; Majority_loss {};'.format(epoch + 1, avg_minority_loss.item(), avg_majority_loss.item()))
            results0, results1, results2, results3 = self.test(self.val_loader)
            mark = recorder.add(results0)
            with open(log_file, "a") as f:
                f.write(f"Epoch {epoch + 1} Finished, Average Loss: {avg_loss.item()}, Minority_Loss: {avg_minority_loss.item()}, Majority_loss: {avg_majority_loss.item()};\n")
            print('result')
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_weibo_new.pkl'))
            elif mark == 'esc':
                break
            else:
                continue

        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_weibo_new.pkl')))
        results0,results1,results2,results3 = self.test(self.val_loader)
        print(results0)
        return results0, os.path.join(self.save_param_dir, 'parameter_weibo_new.pkl')


    def test(self, dataloader):
        pred0 = []
        pred1 = []
        pred2 = []
        pred3 = []
        label1 = []
        category = []
        final_label_pred_sne = [[] for _ in range(9)]
        final_label_label_sne = [[] for _ in range(9)]
        features_by_domain = [[] for _ in range(9)]
        labels_by_domain = [[] for _ in range(9)]
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                
                batch_data = clipdata2gpu(batch)
                batch_size_actual = len(batch_data['content'])

                if batch_size_actual < dataloader.batch_size:
                    batch_needed = dataloader.batch_size - batch_size_actual
                    print(f"not enough, add {batch_needed} data")

                    remaining_data_iter = iter(dataloader)

                    additional_data = next(remaining_data_iter)
                    
                    additional_data_batch = {
                        key: value[:batch_needed].cuda() for key, value in zip(
                            ['content', 'content_masks', 'label', 'category', 'image', 'clip_image', 'clip_text'],
                            additional_data
                        )
                    }

                    for key in batch_data:
                        batch_data[key] = torch.cat([batch_data[key]] + [additional_data_batch[key]], dim=0)

                    print(f"not enough, add {batch_needed} data")
                    

                label = batch_data['label']
                batch_category = batch_data['category']
                original_features = batch_data['content']

                for i in range(9):
                    domain_mask = (batch_category == i)
                    features_by_domain[i].append(original_features[domain_mask])
                    labels_by_domain[i].append(label[domain_mask])
                
                final_label_pred_list,fusion_label_pred_list,image_label_pred_list,text_label_pred_list = self.model(**batch_data)
                
                idxs = torch.tensor([index for index in batch_category]).view(-1, 1)
                batch_label_pred0 = final_label_pred_list
                batch_label_pred1 = fusion_label_pred_list
                batch_label_pred2 = image_label_pred_list
                batch_label_pred3 = text_label_pred_list

                batch_label = torch.cat([
                                    label[idxs.squeeze() == i] 
                                    for i in range(9)
                                ])
                batch_category = torch.sort(batch_category).values
                label1.extend(batch_label.detach().cpu().numpy().tolist())
                pred0.extend(batch_label_pred0.detach().cpu().numpy().tolist())
                pred1.extend(batch_label_pred1.detach().cpu().numpy().tolist())
                pred2.extend(batch_label_pred2.detach().cpu().numpy().tolist())
                pred3.extend(batch_label_pred3.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metricsTrueFalse(label1, pred0, step_n, category, self.category_dict),metricsTrueFalse(label1, pred1, step_n, category, self.category_dict),metricsTrueFalse(label1, pred2, step_n, category, self.category_dict), metricsTrueFalse(label1, pred3, step_n, category, self.category_dict)
