import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.nn.functional import normalize as normalize
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

class AVSModelClass(nn.Module):
    def __init__(self):
        super(AVSModelClass, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet50(pretrained = True).children())[:-2])
        self.avgpool = nn.AvgPool2d(7)
        self.linear1 = nn.Linear(2048, 512)
        self.linear_cls = nn.Linear(512, 23)
        self.relu = nn.ReLU()
        for params in self.resnet.parameters():
            params.requires_grad = False

    def forward(self, image):

        # image = image.unsqueeze(0)
        x = self.resnet(image)

        # [512, 7, 7] to [512, 1, 1] average pooling
        x = self.avgpool(x).squeeze(3).squeeze(2)

        x_fc = self.linear1(x)

        result = self.linear_cls(x_fc)
        # result = self.relu(x_fc1)
        
        # \assert False
        result = result.view(result.size(0), -1)
        # result = F.softmax(result, dim=1)

        # # Compute the minimum and maximum values along dimension 0
        # min_vals, _ = torch.min(result, dim=0)
        # max_vals, _ = torch.max(result, dim=0)

        # # Perform batch min-max normalization
        # normalized_result = (result - min_vals) / (max_vals - min_vals)
        return result