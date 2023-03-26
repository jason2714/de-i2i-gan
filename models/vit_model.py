import torch

from models.base_model import BaseModel
from models.networks.discriminator import ViTClassifier
from torch import nn
from transformers import ViTForImageClassification


class ViTModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.netC = ViTClassifier(opt.label_nc).to(opt.device)
        self.model_ViT = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                                   output_hidden_states=True).to(opt.device)
        self.model_ViT.eval()
        for param in self.model_ViT.parameters():
            param.requires_grad = False
        if self.opt.is_train or hasattr(opt, 'clf_loss_type'):
            assert opt.clf_loss_type is not None, 'clf_loss_type should be initialized in dataset'
            self.clf_loss_type = opt.clf_loss_type

    def __call__(self, data, labels, inference=False):
        if inference:
            self.netC.eval()
            with torch.no_grad():
                return self._compute_classifier_loss(data, labels)
        else:
            self.netC.train()
            return self._compute_classifier_loss(data, labels)

    def _compute_classifier_loss(self, data, labels):
        data, labels = data.to(self.netC.device, non_blocking=True), \
                             labels.to(self.netC.device, non_blocking=True)
        inputs = {'pixel_values': data}
        out_vit = self.model_ViT(**inputs)
        embeddings = out_vit.hidden_states[-1][:, 0, :]
        logits = self.netC(embeddings)
        clf_loss = self._cal_loss(logits, labels, self.clf_loss_type)
        return logits, clf_loss
