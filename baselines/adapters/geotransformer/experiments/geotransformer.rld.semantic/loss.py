import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.loss import WeightedCircleLoss


class CoarseMatchingLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )

        self.positive_overlap = (
            cfg.coarse_loss.positive_overlap
        )

    def forward(self, output_dict):

        ref_feats = output_dict["ref_feats_c"]
        src_feats = output_dict["src_feats_c"]

        gt_indices = (
            output_dict["gt_node_corr_indices"]
        )

        gt_overlaps = (
            output_dict["gt_node_corr_overlaps"]
        )

        ref_gt = gt_indices[:, 0]
        src_gt = gt_indices[:, 1]

        feat_dists = torch.sqrt(
            pairwise_distance(
                ref_feats,
                src_feats,
                normalized=True,
            )
        )

        overlaps = torch.zeros_like(feat_dists)

        overlaps[
            ref_gt,
            src_gt,
        ] = gt_overlaps

        pos_masks = (
            overlaps > self.positive_overlap
        )

        neg_masks = overlaps == 0

        pos_scales = torch.sqrt(
            overlaps * pos_masks.float()
        )

        return self.weighted_circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            pos_scales,
        )


class FineMatchingLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()

    def forward(
        self,
        output_dict,
        data_dict,
    ):

        ref_ids = (
            output_dict[
                "ref_node_corr_knn_ids"
            ]
        )

        src_ids = (
            output_dict[
                "src_node_corr_knn_ids"
            ]
        )

        ref_masks = (
            output_dict[
                "ref_node_corr_knn_masks"
            ]
        )

        src_masks = (
            output_dict[
                "src_node_corr_knn_masks"
            ]
        )

        scores = output_dict["matching_scores"]

        valid = torch.logical_and(
            ref_masks.unsqueeze(2),
            src_masks.unsqueeze(1),
        )

        valid = torch.logical_and(
            valid,
            ref_ids.unsqueeze(2) >= 0,
        )

        valid = torch.logical_and(
            valid,
            src_ids.unsqueeze(1) >= 0,
        )

        gt_corr = (
            ref_ids.unsqueeze(2)
            == src_ids.unsqueeze(1)
        )

        gt_corr = torch.logical_and(
            gt_corr,
            valid,
        )

        # dustbin labels for unmatched neurons
        slack_rows = torch.logical_and(
            gt_corr.sum(dim=2) == 0,
            ref_masks,
        )

        slack_cols = torch.logical_and(
            gt_corr.sum(dim=1) == 0,
            src_masks,
        )

        labels = torch.zeros_like(
            scores,
            dtype=torch.bool,
        )

        labels[:, :-1, :-1] = gt_corr
        labels[:, :-1, -1] = slack_rows
        labels[:, -1, :-1] = slack_cols

        if labels.sum() == 0:
            return scores.sum() * 0.0

        return -scores[labels].mean()


class OverallLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)

        self.weight_coarse_loss = (
            cfg.loss.weight_coarse_loss
        )

        self.weight_fine_loss = (
            cfg.loss.weight_fine_loss
        )

    def forward(
        self,
        output_dict,
        data_dict,
    ):

        coarse_loss = self.coarse_loss(
            output_dict
        )

        fine_loss = self.fine_loss(
            output_dict,
            data_dict,
        )

        loss = (
            self.weight_coarse_loss
            * coarse_loss
            +
            self.weight_fine_loss
            * fine_loss
        )

        return {
            "loss": loss,
            "c_loss": coarse_loss,
            "f_loss": fine_loss,
        }


def build_dense_scores(output_dict):

    ref_ids = output_dict["ref_ids"]
    src_ids = output_dict["src_ids"]

    nr = ref_ids.shape[0]
    ns = src_ids.shape[0]

    device = ref_ids.device

    dense = torch.full(
        (nr, ns),
        -1e9,
        device=device,
        dtype=output_dict[
            "matching_scores"
        ].dtype,
    )

    ref_indices = (
        output_dict[
            "ref_node_corr_knn_indices"
        ]
    )

    src_indices = (
        output_dict[
            "src_node_corr_knn_indices"
        ]
    )

    ref_masks = (
        output_dict[
            "ref_node_corr_knn_masks"
        ]
    )

    src_masks = (
        output_dict[
            "src_node_corr_knn_masks"
        ]
    )

    scores = output_dict[
        "matching_scores"
    ][:, :-1, :-1]

    for p in range(scores.shape[0]):

        rm = ref_masks[p]
        sm = src_masks[p]

        ri = ref_indices[p][rm]
        si = src_indices[p][sm]

        if ri.numel() == 0 or si.numel() == 0:
            continue

        patch = scores[p][rm][:, sm]

        old = dense[
            ri[:, None],
            si[None, :],
        ]

        dense[
            ri[:, None],
            si[None, :],
        ] = torch.maximum(
            old,
            patch,
        )

    return dense


class Evaluator(nn.Module):

    def __init__(self, cfg):
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        output_dict,
        data_dict,
    ):

        dense = build_dense_scores(
            output_dict
        )

        ref_ids = output_dict["ref_ids"]
        src_ids = output_dict["src_ids"]

        gt_map = (
            ref_ids[:, None]
            == src_ids[None, :]
        )

        gt_map = torch.logical_and(
            gt_map,
            ref_ids[:, None] >= 0,
        )

        gt_map = torch.logical_and(
            gt_map,
            src_ids[None, :] >= 0,
        )

        valid_q = gt_map.any(dim=1)

        num_q = valid_q.sum()

        if num_q == 0:

            z = dense.sum() * 0.0

            return {
                "Top1": z,
                "Top5": z,
                "MRR": z,
            }

        gt_index = gt_map.float().argmax(dim=1)

        pred1 = dense.argmax(dim=1)

        row_has_candidate = (
            dense.max(dim=1).values > -1e8
        )

        top1_ok = torch.logical_and(
            pred1 == gt_index,
            row_has_candidate,
        )

        top1 = (
            top1_ok[valid_q]
            .float()
            .mean()
        )

        k = min(
            5,
            dense.shape[1],
        )

        topk = torch.topk(
            dense,
            k=k,
            dim=1,
        ).indices

        top5_ok = (
            topk
            == gt_index[:, None]
        ).any(dim=1)

        top5_ok = torch.logical_and(
            top5_ok,
            row_has_candidate,
        )

        top5 = (
            top5_ok[valid_q]
            .float()
            .mean()
        )

        rows = torch.arange(
            dense.shape[0],
            device=dense.device,
        )

        target_score = dense[
            rows,
            gt_index,
        ]

        target_seen = target_score > -1e8

        rank = 1 + (
            dense
            > target_score[:, None]
        ).sum(dim=1)

        rr = torch.zeros(
            dense.shape[0],
            device=dense.device,
            dtype=dense.dtype,
        )

        good = torch.logical_and(
            valid_q,
            target_seen,
        )

        rr[good] = (
            1.0
            / rank[good].float()
        )

        mrr = rr[valid_q].mean()

        return {
            "Top1": top1,
            "Top5": top5,
            "MRR": mrr,
        }
