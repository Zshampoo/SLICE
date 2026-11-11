import torch
from torch import nn
from pathlib import Path
import tempfile
from modules import MLP
from torchvision.models import resnet50, ResNet50_Weights
# from LLAVA.biovil_t.pretrained import get_biovil_t_image_encoder
JOINT_FEATURE_SIZE = 128


class ChexpertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder, feature_size = self.get_encoder()

        self.projector = MLP(input_dim=feature_size, output_dim=JOINT_FEATURE_SIZE, hidden_dim=JOINT_FEATURE_SIZE, use_1x1_convs=True)

        # MLP
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        # FC
        # self.fc = nn.Linear(128 * 4 * 4, num_classes)
    def get_encoder(self):
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # since we have grayscale images, we need to change the first conv layer to accept 1 in_channel (instead of 3)
        # resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # use only the feature extractor of the pre-trained classification model
        # (i.e. use all children but the last 2, which are AdaptiveAvgPool2d and Linear)
        backbone = nn.Sequential(*list(resnet.children())[:-2])

        out_channels = 2048

        return backbone, out_channels

    def forward(self, x):
        # encoding
        x = self.encoder(x) # (batch, 2048, 16, 16)
        # flatten
        # x = x.view(x.size(0), -1)
        x = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))(x) # (batch, 2048, 1, 1)
        x = x.view(x.shape[0], -1)  # Flatten the tensor # (batch, 2048)

        # projecting
        # x = self.projector(x)

        # classify head
        x = torch.relu(self.fc1(x)) # (batch, 512)
        x = self.fc2(x) # (batch, 14)

        # x = self.fc(x)  # (batch, 14)
        # x = self.biovil_encoder(x).img_embedding
        return x

if __name__ == "__main__":
    model = ChexpertClassifier()