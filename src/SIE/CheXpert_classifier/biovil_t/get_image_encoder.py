from pathlib import Path
import torch
from torch import nn

from src.SIE.CheXpert_classifier.biovil_t.model import ImageModel
from src.SIE.CheXpert_classifier.biovil_t.pretrained import JOINT_FEATURE_SIZE, get_biovil_t_image_encoder
from src.SIE.CheXpert_classifier.biovil_t.types import ImageEncoderType


def get_biovil_image_encoder(pretrained: bool = True) -> ImageModel:
    """Download weights from Hugging Face and instantiate the image model."""
    resnet_checkpoint_path = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))

    image_model = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=resnet_checkpoint_path,
    )
    return image_model

class ChexpertClassifier_(nn.Module):
    def __init__(self, num_classes, head_mode="bce"):
        """
        head_mode:
          - "bce": one linear layer, num_classes logits (multi-label)
          - "chexbert4": 14 independent heads, same layout as bert_labeler
              heads 0-12: 4-way (blank=0, positive=1, negative=2, uncertain=3)
              head 13 No Finding: 2-way (blank=0, positive=1)
        """
        super().__init__()
        # self.biovil_encoder = get_biovil_t_image_encoder()
        self.biovil_encoder = get_biovil_image_encoder()
        self.head_mode = head_mode
        self.num_classes = num_classes

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        hidden = 512
        if head_mode == "chexbert4":
            # 13 conditions x {blank, positive, negative, uncertain}
            self.linear_heads = nn.ModuleList(
                [nn.Linear(hidden, 4, bias=True) for _ in range(13)]
            )
            # No Finding x {blank, positive}
            self.linear_heads.append(nn.Linear(hidden, 2, bias=True))
        else:
            self.fc2 = nn.Linear(hidden, num_classes)

    def encode(self, x):
        x = self.biovil_encoder(x).projected_patch_embeddings
        x = torch.nn.functional.avg_pool2d(x, 4)
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        return torch.relu(self.fc1(x))

    def forward(self, x):
        hidden = self.encode(x)
        if self.head_mode == "chexbert4":
            return [head(hidden) for head in self.linear_heads]
        return self.fc2(hidden)
