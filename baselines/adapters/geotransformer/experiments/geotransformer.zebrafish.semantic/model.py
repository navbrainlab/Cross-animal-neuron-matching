import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.ops import (
    point_to_node_partition,
    index_select,
)

from geotransformer.modules.sinkhorn import (
    LearnableLogOptimalTransport,
)

from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
)

from backbone import KPConvFPN


def semantic_node_correspondences(
    ref_patch_ids,
    src_patch_ids,
    ref_masks,
    src_masks,
):
    """
    Build node-level GT from shared neuron identities.

    ref_patch_ids: [R, K]
    src_patch_ids: [S, K]
    """

    ref_valid = torch.logical_and(
        ref_masks,
        ref_patch_ids >= 0,
    )

    src_valid = torch.logical_and(
        src_masks,
        src_patch_ids >= 0,
    )

    # [R, S, Kr, Ks]
    equal = (
        ref_patch_ids[:, None, :, None]
        == src_patch_ids[None, :, None, :]
    )

    equal = torch.logical_and(
        equal,
        ref_valid[:, None, :, None],
    )

    equal = torch.logical_and(
        equal,
        src_valid[None, :, None, :],
    )

    # fraction of labelled neurons in each patch
    # that also occur in the opposite patch
    ref_shared = equal.any(dim=-1).sum(dim=-1).float()
    src_shared = equal.any(dim=-2).sum(dim=-1).float()

    ref_count = (
        ref_valid.sum(dim=-1)
        .float()
        .clamp_min(1.0)
    )[:, None]

    src_count = (
        src_valid.sum(dim=-1)
        .float()
        .clamp_min(1.0)
    )[None, :]

    overlap = 0.5 * (
        ref_shared / ref_count
        + src_shared / src_count
    )

    positive = overlap > 0

    indices = positive.nonzero(as_tuple=False)

    if indices.numel() == 0:
        raise RuntimeError(
            "No semantic node correspondence found."
        )

    overlaps = overlap[
        indices[:, 0],
        indices[:, 1],
    ]

    return indices, overlaps


class GeoTransformer(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.num_points_in_patch = (
            cfg.model.num_points_in_patch
        )

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets,
            cfg.coarse_matching.overlap_threshold,
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences,
            cfg.coarse_matching.dual_normalization,
        )

        self.optimal_transport = (
            LearnableLogOptimalTransport(
                cfg.model.num_sinkhorn_iterations
            )
        )

    def forward(self, data_dict):

        output_dict = {}

        feats = data_dict["features"].detach()

        ref_ids = data_dict["ref_ids"].long()
        src_ids = data_dict["src_ids"].long()

        ref_length_c = (
            data_dict["lengths"][-1][0].item()
        )

        ref_length_f = (
            data_dict["lengths"][0][0].item()
        )

        ref_length = (
            data_dict["lengths"][0][0].item()
        )

        points_c = data_dict["points"][-1].detach()
        points_f = data_dict["points"][0].detach()
        points = data_dict["points"][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]

        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]

        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict.update({
            "ref_points_c": ref_points_c,
            "src_points_c": src_points_c,
            "ref_points_f": ref_points_f,
            "src_points_f": src_points_f,
            "ref_points": ref_points,
            "src_points": src_points,

            "ref_ids": ref_ids,
            "src_ids": src_ids,
        })

        # --------------------------------------------------------------
        # 1. Fine points -> coarse nodes
        # --------------------------------------------------------------
        (
            _,
            ref_node_masks,
            ref_node_knn_indices,
            ref_node_knn_masks,
        ) = point_to_node_partition(
            ref_points_f,
            ref_points_c,
            self.num_points_in_patch,
        )

        (
            _,
            src_node_masks,
            src_node_knn_indices,
            src_node_knn_masks,
        ) = point_to_node_partition(
            src_points_f,
            src_points_c,
            self.num_points_in_patch,
        )

        ref_padded_points_f = torch.cat(
            [
                ref_points_f,
                torch.zeros_like(ref_points_f[:1]),
            ],
            dim=0,
        )

        src_padded_points_f = torch.cat(
            [
                src_points_f,
                torch.zeros_like(src_points_f[:1]),
            ],
            dim=0,
        )

        ref_node_knn_points = index_select(
            ref_padded_points_f,
            ref_node_knn_indices,
            dim=0,
        )

        src_node_knn_points = index_select(
            src_padded_points_f,
            src_node_knn_indices,
            dim=0,
        )

        # --------------------------------------------------------------
        # Semantic IDs for node patches
        # --------------------------------------------------------------
        ref_padded_ids = torch.cat(
            [
                ref_ids,
                ref_ids.new_full((1,), -1),
            ],
            dim=0,
        )

        src_padded_ids = torch.cat(
            [
                src_ids,
                src_ids.new_full((1,), -1),
            ],
            dim=0,
        )

        ref_node_knn_ids = (
            ref_padded_ids[ref_node_knn_indices]
        )

        src_node_knn_ids = (
            src_padded_ids[src_node_knn_indices]
        )

        # --------------------------------------------------------------
        # 2. Semantic GT node correspondences
        # --------------------------------------------------------------
        (
            gt_node_corr_indices,
            gt_node_corr_overlaps,
        ) = semantic_node_correspondences(
            ref_node_knn_ids,
            src_node_knn_ids,
            ref_node_knn_masks,
            src_node_knn_masks,
        )

        output_dict[
            "gt_node_corr_indices"
        ] = gt_node_corr_indices

        output_dict[
            "gt_node_corr_overlaps"
        ] = gt_node_corr_overlaps

        # --------------------------------------------------------------
        # 3. KPConv encoder
        # --------------------------------------------------------------
        feats_list = self.backbone(
            feats,
            data_dict,
        )

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # --------------------------------------------------------------
        # 4. Official Geometric Transformer
        # --------------------------------------------------------------
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]

        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )

        ref_feats_c = F.normalize(
            ref_feats_c.squeeze(0),
            p=2,
            dim=1,
        )

        src_feats_c = F.normalize(
            src_feats_c.squeeze(0),
            p=2,
            dim=1,
        )

        output_dict["ref_feats_c"] = ref_feats_c
        output_dict["src_feats_c"] = src_feats_c

        # --------------------------------------------------------------
        # Fine features
        # --------------------------------------------------------------
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]

        output_dict["ref_feats_f"] = ref_feats_f
        output_dict["src_feats_f"] = src_feats_f

        # --------------------------------------------------------------
        # 5. Predicted coarse matches
        # --------------------------------------------------------------
        with torch.no_grad():

            (
                pred_ref_node_indices,
                pred_src_node_indices,
                pred_node_scores,
            ) = self.coarse_matching(
                ref_feats_c,
                src_feats_c,
                ref_node_masks,
                src_node_masks,
            )

            output_dict[
                "ref_node_corr_indices"
            ] = pred_ref_node_indices

            output_dict[
                "src_node_corr_indices"
            ] = pred_src_node_indices

            if self.training:
                (
                    ref_node_corr_indices,
                    src_node_corr_indices,
                    node_corr_scores,
                ) = self.coarse_target(
                    gt_node_corr_indices,
                    gt_node_corr_overlaps,
                )
            else:
                ref_node_corr_indices = (
                    pred_ref_node_indices
                )
                src_node_corr_indices = (
                    pred_src_node_indices
                )
                node_corr_scores = pred_node_scores

        output_dict[
            "used_ref_node_corr_indices"
        ] = ref_node_corr_indices

        output_dict[
            "used_src_node_corr_indices"
        ] = src_node_corr_indices

        # --------------------------------------------------------------
        # 6. Corresponding node patches
        # --------------------------------------------------------------
        ref_patch_indices = (
            ref_node_knn_indices[
                ref_node_corr_indices
            ]
        )

        src_patch_indices = (
            src_node_knn_indices[
                src_node_corr_indices
            ]
        )

        ref_patch_masks = (
            ref_node_knn_masks[
                ref_node_corr_indices
            ]
        )

        src_patch_masks = (
            src_node_knn_masks[
                src_node_corr_indices
            ]
        )

        ref_patch_points = (
            ref_node_knn_points[
                ref_node_corr_indices
            ]
        )

        src_patch_points = (
            src_node_knn_points[
                src_node_corr_indices
            ]
        )

        ref_patch_ids = (
            ref_node_knn_ids[
                ref_node_corr_indices
            ]
        )

        src_patch_ids = (
            src_node_knn_ids[
                src_node_corr_indices
            ]
        )

        ref_padded_feats_f = torch.cat(
            [
                ref_feats_f,
                torch.zeros_like(
                    ref_feats_f[:1]
                ),
            ],
            dim=0,
        )

        src_padded_feats_f = torch.cat(
            [
                src_feats_f,
                torch.zeros_like(
                    src_feats_f[:1]
                ),
            ],
            dim=0,
        )

        ref_patch_feats = index_select(
            ref_padded_feats_f,
            ref_patch_indices,
            dim=0,
        )

        src_patch_feats = index_select(
            src_padded_feats_f,
            src_patch_indices,
            dim=0,
        )

        # --------------------------------------------------------------
        # 7. Sinkhorn fine matching
        # --------------------------------------------------------------
        matching_scores = torch.einsum(
            "bnd,bmd->bnm",
            ref_patch_feats,
            src_patch_feats,
        )

        matching_scores = (
            matching_scores
            / (feats_f.shape[1] ** 0.5)
        )

        matching_scores = self.optimal_transport(
            matching_scores,
            ref_patch_masks,
            src_patch_masks,
        )

        output_dict.update({
            "matching_scores": matching_scores,

            "ref_node_corr_knn_indices":
                ref_patch_indices,

            "src_node_corr_knn_indices":
                src_patch_indices,

            "ref_node_corr_knn_masks":
                ref_patch_masks,

            "src_node_corr_knn_masks":
                src_patch_masks,

            "ref_node_corr_knn_points":
                ref_patch_points,

            "src_node_corr_knn_points":
                src_patch_points,

            "ref_node_corr_knn_ids":
                ref_patch_ids,

            "src_node_corr_knn_ids":
                src_patch_ids,
        })

        return output_dict


def create_model(cfg):
    return GeoTransformer(cfg)
