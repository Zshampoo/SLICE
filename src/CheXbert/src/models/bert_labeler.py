from pathlib import Path
import torch.nn as nn
from transformers import BertModel, AutoModel
BERT_PATH = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))

# Chest ImaGenome 29 anatomy regions (column order in sentence_region_label_v2)
REGION_NAMES_29 = [
    'right lung', 'right upper lung zone', 'right mid lung zone', 'right lower lung zone',
    'right hilar structures', 'right apical zone', 'right costophrenic angle', 'right hemidiaphragm',
    'left lung', 'left upper lung zone', 'left mid lung zone', 'left lower lung zone',
    'left hilar structures', 'left apical zone', 'left costophrenic angle', 'left hemidiaphragm',
    'trachea', 'spine', 'right clavicle', 'left clavicle', 'aortic arch', 'mediastinum',
    'upper mediastinum', 'svc', 'cardiac silhouette', 'cavoatrial junction', 'right atrium',
    'carina', 'abdomen',
]

# blank / normal / abnormal for sentence-region task
REGION_LABEL2ID = {'blank': 0, 'normal': 1, 'abnormal': 2}
REGION_ID2LABEL = {0: 'blank', 1: 'normal', 2: 'abnormal'}

# report-level region binary: normal=0, abnormal=1 (no blank)
REPORT_REGION_LABEL2ID = {'normal': 0, 'abnormal': 1}
REPORT_REGION_ID2LABEL = {0: 'normal', 1: 'abnormal'}


class bert_labeler(nn.Module):
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None,
                 task='chexbert', n_regions=29, n_classes=3):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        @param task (str): 'chexbert' | 'sentence_region' (29×3) | 'report_region' (29×2)
        @param n_regions (int): number of region heads for region tasks
        @param n_classes (int): classes per region head
            - sentence_region: blank=0, normal=1, abnormal=2 (default 3)
            - report_region: normal=0, abnormal=1 (default overridden to 2)
        """
        super(bert_labeler, self).__init__()

        self.task = task
        self.n_regions = n_regions
        if task == 'report_region' and n_classes == 3:
            n_classes = 2
        self.n_classes = n_classes

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.bert = BertModel.from_pretrained(BERT_PATH)

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p)
        # size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features

        if task in ('sentence_region', 'report_region'):
            # sentence_region: 29 × {blank, normal, abnormal}
            # report_region:   29 × {normal, abnormal} binary heads
            self.linear_heads = nn.ModuleList(
                [nn.Linear(hidden_size, n_classes, bias=True) for _ in range(n_regions)]
            )
        else:
            # classes: present, absent, unknown, blank for 12 conditions + support devices
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
            # classes: yes, no for the 'no finding' observation
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def get_cls_hidden_state(self, source_padded, attention_mask):
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        # shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        return cls_hidden

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor]):
            - chexbert: 14 tensors (13×(B,4) + 1×(B,2))
            - sentence_region: n_regions tensors each (B, 3)
            - report_region: n_regions tensors each (B, 2)
        """
        # shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        # shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(len(self.linear_heads)):
            out.append(self.linear_heads[i](cls_hidden))
        return out
