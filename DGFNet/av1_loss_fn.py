from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import gpu, to_long


class LossFunc(nn.Module):
    def __init__(self, config, device):
        super(LossFunc, self).__init__()
        self.config = config
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out_sc, data):

        loss_out = self.pred_loss(
                                  out_sc,
                                  gpu(data["TRAJS_FUT"], self.device),
                                  to_long(gpu(data["PAD_FUT"], self.device)),
                                  gpu(data["TRAJS_FUT_ORI"], self.device),
                                  )
        loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"] + loss_out["reg_loss_final"]
        return loss_out

    def pred_loss(self, out_sc: Dict[str, List[torch.Tensor]], gt_preds: List[torch.Tensor], pad_flags: List[torch.Tensor], gt_preds_sc: List[torch.Tensor]):
        cls, reg, reg_final = map(lambda x: torch.cat(x, 0), out_sc[:3])
        gt_preds = torch.cat(gt_preds, 0)
        has_preds = torch.cat(pad_flags, 0).bool()

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = 30

        mask, last_idcs = self.create_mask(has_preds, num_preds)
        cls, reg, reg_final, gt_preds, has_preds, last_idcs = map(lambda x: x[mask], [cls, reg, reg_final, gt_preds, has_preds, last_idcs])

        dist, min_dist, min_idcs = self.get_min_distance_indices(reg[..., 0:2].clone(), gt_preds, last_idcs, num_modes)
        cls_loss = self.calculate_classification_loss(cls, min_idcs, mask, dist, min_dist)
        reg_loss = self.calculate_regression_loss(reg, min_idcs, gt_preds, has_preds)
        reg_loss_final = self.calculate_regression_loss(reg_final[..., 0:2].clone(), min_idcs, gt_preds, has_preds)

        loss_out["cls_loss"] = self.config["cls_coef"] * cls_loss
        loss_out["reg_loss"] = self.config["reg_coef"] * reg_loss
        loss_out["reg_loss_final"] = self.config["reg_coef_final"] * reg_loss_final

        return loss_out

    def create_mask(self, has_preds, num_preds):
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0
        return mask, last_idcs

    def get_min_distance_indices(self, reg, gt_preds, last_idcs, num_modes):
        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)
        dist = [torch.sqrt(((reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs]) ** 2).sum(1)) for j in range(num_modes)]
        dist = torch.stack(dist, dim=1)
        min_dist, min_idcs = dist.min(1)
        return dist, min_dist, min_idcs
    
    def calculate_classification_loss(self, cls, min_idcs, mask, dist, min_dist):
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        num_cls = mask.sum().item()
        cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)
        return cls_loss

    def calculate_regression_loss(self, reg, min_idcs, gt_preds, has_preds):
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        reg = reg[row_idcs, min_idcs]
        num_reg = has_preds.sum().item()
        reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (num_reg + 1e-10)
        return reg_loss

