import torch
from torch import nn
from torchvision.models import mobilenet_v2

class GeneratorLoss(nn.Module):
    """Generator loss combining adversarial, perceptual, image, and TV losses."""
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = mobilenet_v2(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        adversarial_loss = torch.mean(1 - out_labels)
        with torch.no_grad():
            perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        image_loss = self.mse_loss(out_images, target_images)
        tv_loss = self.tv_loss(out_images)
        total_loss = image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
        return total_loss

class TVLoss(nn.Module):
    """Total Variation loss."""
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size, h_x, w_x = x.size(0), x.size(2), x.size(3)
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        tv_loss = self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        return tv_loss

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
