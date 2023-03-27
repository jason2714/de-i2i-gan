import torch

from models.base_model import BaseModel
from models.networks.discriminator import ViTClassifier
from torch import nn
from transformers import ViTForImageClassification


class ViTModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.netC = ViTClassifier(opt.label_nc).to(opt.device)
        self.netViT_ = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                                 output_hidden_states=True).to(opt.device)

        # self._netViT_ = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
        #                                                            output_hidden_states=True).to(opt.device)
        # self._netViT_.eval()
        # for param in self._netViT_.parameters():
        #     param.requires_grad = False

        if self.opt.is_train or hasattr(opt, 'clf_loss_type'):
            assert opt.clf_loss_type is not None, 'clf_loss_type should be initialized in dataset'
            self.clf_loss_type = opt.clf_loss_type

    def __call__(self, mode, data, labels=None):
        if mode == 'inference':
            self.netC.eval()
            self.netViT_.eval()
            with torch.no_grad():
                return self._compute_classifier_loss(data, labels)
        elif mode == 'train':
            self.netC.train()
            self.netViT_.train()
            return self._compute_classifier_loss(data, labels)
        elif mode == 'get_embedding':
            self.netC.eval()
            self.netViT_.eval()
            with torch.no_grad():
                return self._get_embedding(data)

    def _compute_classifier_loss(self, data, labels):
        data, labels = data.to(self.netC.device, non_blocking=True), \
                       labels.to(self.netC.device, non_blocking=True)
        inputs = {'pixel_values': data}
        out_vit = self.netViT_(**inputs)
        embeddings = out_vit.hidden_states[-1][:, 0, :]
        logits = self.netC(embeddings)
        clf_loss = self._cal_loss(logits, labels, self.clf_loss_type)
        return logits, clf_loss

    def _get_embedding(self, data):
        data = data.to(self.netC.device, non_blocking=True)
        inputs = {'pixel_values': data}
        out_vit = self.netViT_(**inputs)
        embeddings = out_vit.hidden_states[-1][:, 0, :]
        return embeddings
