import torch
import torch.nn as nn

class Norm_L2_Loss(nn.Module):
    def __init__(self, cfg):
        super(Norm_L2_Loss, self).__init__()
        self.flag = cfg.DATASET.DATASET

    def forward(self, input_tensor, ground_truth, mask):
        L2_Loss = torch.mean(torch.norm((input_tensor - ground_truth), dim=2) * mask, dim=1)
        if ground_truth.size(1) == 98:
            L2_norm = torch.norm(ground_truth[:, 60, :] - ground_truth[:, 72, :], dim=1)
        elif ground_truth.size(1) == 68:
            L2_norm = torch.norm(ground_truth[:, 36, :] - ground_truth[:, 45, :], dim=1)
        elif ground_truth.size(1) == 29:
            L2_norm = torch.norm(ground_truth[:, 8, :] - ground_truth[:, 9, :], dim=1)
        else:
            raise NotImplementedError
        L2_Loss = L2_Loss / L2_norm
        return torch.mean(L2_Loss)


class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss, self).__init__()

    def forward(self, input_tensor, ground_truth, mask):
        L1_Loss = torch.mean(torch.mean(torch.abs((input_tensor - ground_truth)), dim=-1) * mask)
        return L1_Loss