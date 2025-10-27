# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead
from .standard_roi_head import StandardRoIHead

from pathlib import Path

@MODELS.register_module()
class IcdaRoIHead(StandardRoIHead): # method
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 task_id: int,
                 ufm_mem_path: str,
                 ufm_mem_length: int,
                 num_heads: int,
                 num_attn_layers: int,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:


        c, w, h = bbox_roi_extractor['out_channels'], bbox_roi_extractor['roi_layer']['output_size'], bbox_roi_extractor['roi_layer']['output_size']
        super().__init__(
            bbox_roi_extractor = bbox_roi_extractor,
            bbox_head = bbox_head,
            mask_roi_extractor = mask_roi_extractor,
            mask_head = mask_head,
            shared_head = shared_head,
            train_cfg = train_cfg,
            test_cfg = test_cfg,
            init_cfg = init_cfg)


        self.task_id = task_id
        self.ufm_mem_path = ufm_mem_path
        self.num_classes = self.bbox_head.num_classes
        self.ufm_mem_length = ufm_mem_length

        self.cur_ufm = [[] for _ in range(self.num_classes)]
        self.prev_ufm = []

        for i in range(self.task_id):
            mem_file = Path(self.ufm_mem_path) / f"task_{i}_mem.pt"
            self.prev_ufm.append(torch.load(mem_file))

        self.prev_ufm = [torch.cat([torch.cat(tensor_group, dim=0) for tensor_group in tensor_groups], dim=0) for tensor_groups in zip(*self.prev_ufm)]


        if self.task_id != 0:
            self.attn_layers = torch.nn.ModuleList()
            for _ in range(num_attn_layers):
                attn_layer = torch.nn.MultiheadAttention(
                    embed_dim= c * w * h,
                    num_heads=num_heads,
                    batch_first=True
                )
                self.attn_layers.append(attn_layer)


    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor, labels, bg_class_id) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)


        attn_embeds = bbox_feats.clone()
        bs = bbox_feats.shape[0]


        if self.training:

            for idx in range(bs):
                label = labels[idx]

                if label != bg_class_id:
                    current_label = label.item()


                    bbox_feat_copy = bbox_feats[idx].flatten(0).clone().detach().unsqueeze(0)
                    self.cur_ufm[current_label].append(bbox_feat_copy)

                    if len(self.cur_ufm[current_label]) > self.ufm_mem_length:
                        features_tensor = torch.cat(self.cur_ufm[current_label], dim=0)
                        features_norm = F.normalize(features_tensor, p=2, dim=1)
                        cos_sim_matrix = torch.matmul(features_norm, features_norm.t())
                        sim_sums = torch.sum(cos_sim_matrix, dim=1)
                        index_to_remove = torch.argmax(sim_sums)
                        self.cur_ufm[current_label].pop(index_to_remove.item())


                    if len(self.prev_ufm) > 0:
                        prev_features = self.prev_ufm[current_label]
                        if prev_features.shape[0] > 0:
                            # query: (C, H, W) -> (D_flat) -> (1, 1, D_flat)
                            query = bbox_feats[idx].flatten(0).clone().unsqueeze(0).unsqueeze(1)
                            # key/value: (K, D_flat) -> (1, K, D_flat)
                            key = prev_features.unsqueeze(0)
                            value = prev_features.unsqueeze(0)

                            for attn_layer in self.attn_layers:
                                query, _ = attn_layer(query, key, value)

                            attn_embeds[idx] = query.view_as(bbox_feats[idx])


            torch.save(self.cur_ufm, Path(self.ufm_mem_path) / f"task_{self.task_id}_mem.pt")

        else:
            has_prev_memory = len(self.prev_ufm) > 0
            if has_prev_memory:
                normalized_memory = {}
                for label, features in self.prev_ufm.items():
                    if features.shape[0] > 0:
                        normalized_memory[label] = F.normalize(features, p=2, dim=1)

                if not normalized_memory:
                    pass
                else:
                    for idx in range(bs):
                        # (C, H, W) -> (D_flat) -> (1, D)
                        current_feat_flat = bbox_feats[idx].flatten(0).unsqueeze(0)
                        current_feat_norm = F.normalize(current_feat_flat, p=2, dim=1)

                        best_label = -1
                        best_avg_sim = -torch.inf
                        for label, mem_norm in normalized_memory.items():
                            cos_sims = torch.matmul(current_feat_norm, mem_norm.t())  # (1, K)
                            avg_sim = torch.mean(cos_sims)
                            if avg_sim > best_avg_sim:
                                best_avg_sim = avg_sim
                                best_label = label


                        prev_features = self.prev_ufm[best_label]

                        # query: (1, D) -> (1, 1, D)
                        query = current_feat_flat.clone().unsqueeze(1)
                        # key/value: (K, D) -> (1, K, D)
                        key = prev_features.unsqueeze(0)
                        value = prev_features.unsqueeze(0)

                        for attn_layer in self.attn_layers:
                            query, _ = attn_layer(query, key, value)

                        attn_embeds[idx] = query.squeeze(1).view_as(bbox_feats[idx])

        bbox_feats = torch.cat([bbox_feats, attn_embeds], dim=1)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])

        labels, _, _, _ = self.bbox_head.get_targets(
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)
        bg_class_id = labels[-1]

        bbox_results = self._bbox_forward(x, rois, labels, bg_class_id)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois, labels=None, bg_class_id=None)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list
