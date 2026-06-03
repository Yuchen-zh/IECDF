import torch
import torch.nn as nn
import torch.nn.functional as F

class DLINEXLoss(nn.Module):
    def __init__(self, a, category_label_distribution, categorys, n_classes, device="cuda"):
        super(DLINEXLoss, self).__init__()
        self.a = a
        self.category_label_distribution = category_label_distribution
        self.category = categorys
        self.n_classes = n_classes
        self.device = device

        self.category_total_counts = {}
        for category_id, counts in self.category_label_distribution.items():
            total_count = counts['label_0'] + counts['label_1']
            self.category_total_counts[category_id] = total_count

        self.sign_matrices = {}
        for category_id in self.category_total_counts:
            total_count = self.category_total_counts[category_id]
            delta = total_count / self.n_classes

            label_0_count = self.category_label_distribution[category_id]['label_0']
            label_1_count = self.category_label_distribution[category_id]['label_1']
            delta = torch.tensor(delta, device=self.device)
            sign_0 = torch.where(
                label_0_count < delta,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-1.0, device=self.device)
            )
            sign_1 = torch.where(
                label_1_count < delta,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-1.0, device=self.device)
            )
            sign_matrix = torch.stack([sign_0, sign_1], dim=0)
            self.sign_matrices[category_id] = sign_matrix

    def linex_loss(self, x):
        return torch.exp(self.a * x) - self.a * x - 1

    def _get_true_class_probs(self, domain_outputs, true_class_indices):
        # MLP 分类头输出 shape [N, 1]（sigmoid 后为正类概率），不是 [N, 2]
        true_class_indices = true_class_indices.reshape(-1)
        if domain_outputs.dim() == 0:
            domain_outputs = domain_outputs.unsqueeze(0)
        if domain_outputs.dim() == 1:
            p1 = domain_outputs
        elif domain_outputs.size(-1) == 1:
            p1 = domain_outputs.squeeze(-1)
        else:
            return domain_outputs.gather(1, true_class_indices.unsqueeze(1)).squeeze(-1)
        return torch.where(true_class_indices == 1, p1, 1 - p1)

    def forward(self, outputs, targets, categories):
        total_loss = 0
        minority_class_loss = 0
        majority_class_loss = 0
        
        for i in range(self.category):
            domain_mask = (categories == i).nonzero(as_tuple=True)[0]

            if domain_mask.numel() == 0:
                continue

            domain_outputs = outputs[domain_mask]
            domain_targets = targets[domain_mask]

            if domain_targets.dim() == 1:
                domain_targets = domain_targets.unsqueeze(0)
            true_class_indices = domain_targets.argmax(dim=1)

            sign_matrix = self.sign_matrices[i]

            true_class_signs = sign_matrix[true_class_indices]

            true_class_probs = self._get_true_class_probs(domain_outputs, true_class_indices)

            z = true_class_signs * (1 - true_class_probs)
            domain_loss = self.linex_loss(z)

            for sign, loss in zip(true_class_signs, domain_loss):
                if sign == 1.0:
                    minority_class_loss += loss
                else:
                    majority_class_loss += loss

            total_loss += torch.mean(domain_loss)
        
        return total_loss, minority_class_loss, majority_class_loss
