import torch
from torch import nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, z_dim, input_dim=3):
        super().__init__()
        self.z_dim = z_dim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.mean_fc1 = nn.Linear(512, 256)
        self.mean_fc2 = nn.Linear(256, 128)
        self.mean_fc3 = nn.Linear(128, z_dim)
        self.mean_fc_bn1 = nn.BatchNorm1d(256)
        self.mean_fc_bn2 = nn.BatchNorm1d(128)

        self.var_fc1 = nn.Linear(512, 256)
        self.var_fc2 = nn.Linear(256, 128)
        self.var_fc3 = nn.Linear(128, z_dim)
        self.var_fc_bn1 = nn.BatchNorm1d(256)
        self.var_fc_bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        mean = F.relu(self.mean_fc_bn1(self.mean_fc1(x)))
        mean = F.relu(self.mean_fc_bn2(self.mean_fc2(mean)))
        mean = self.mean_fc3(mean)

        log_var = F.relu(self.var_fc_bn1(self.var_fc1(x)))
        log_var = F.relu(self.var_fc_bn2(self.var_fc2(log_var)))
        log_var = self.var_fc3(log_var)

        return mean, log_var
