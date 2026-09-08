from __future__ import annotations

import argparse
import collections
import collections.abc
import json
import math
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, cKDTree
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor

# ThinkMatch Python 3.10 compatibility
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

from models.BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import (
    SiameseSConvOnNodes,
    SiameseNodeFeaturesToEdgeFeatures,
)
from models.NGM.gnn import PYGNNLayer
from src.lap_solvers.sinkhorn import Sinkhorn
from src.loss_func import PermutationLoss


def normalize_label(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "-1"}:
        return ""
    return text


@dataclass
class Worm:
    path: Path
    uid: str
    xyz: np.ndarray
    labels: list[str]
    supervised: np.ndarray


def load_worm(path: Path) -> Worm:
    with np.load(path, allow_pickle=True) as d:
        xyz = np.asarray(d["xyz"], dtype=np.float32)[:, :3]

        labels = [
            normalize_label(v)
            for v in np.asarray(d["cell_id"]).reshape(-1)
        ]

        if "labeled_mask" in d.files:
            supervised = np.asarray(
                d["labeled_mask"], dtype=bool
            ).reshape(-1)
        elif "clean_mask" in d.files:
            supervised = np.asarray(
                d["clean_mask"], dtype=bool
            ).reshape(-1)
        else:
            supervised = np.asarray(
                [bool(x) for x in labels], dtype=bool
            )

        if "recording_uid" in d.files:
            uid = str(
                np.asarray(d["recording_uid"]).reshape(-1)[0]
            )
        else:
            uid = path.stem

    if len(xyz) != len(labels):
        raise ValueError(f"xyz/label mismatch: {path}")

    if len(labels) != len(supervised):
        raise ValueError(f"mask/label mismatch: {path}")

    if not np.isfinite(xyz).all():
        raise ValueError(f"non-finite xyz: {path}")

    return Worm(
        path=path,
        uid=uid,
        xyz=xyz,
        labels=labels,
        supervised=supervised,
    )


def unique_identity_map(worm: Worm):
    counts = Counter(
        label
        for label, keep in zip(worm.labels, worm.supervised)
        if keep and label
    )

    return {
        label: i
        for i, (label, keep)
        in enumerate(zip(worm.labels, worm.supervised))
        if keep and label and counts[label] == 1
    }


def fit_train_scaler(train_worms):
    all_xyz = np.concatenate(
        [w.xyz for w in train_worms], axis=0
    ).astype(np.float64)

    mean = all_xyz.mean(axis=0)
    std = all_xyz.std(axis=0)

    std[std < 1e-6] = 1.0

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def normalized_xyz(worm, mean, std):
    return (
        (worm.xyz - mean) / std
    ).astype(np.float32)


def symmetric_chamfer(a, b):
    ta = cKDTree(a)
    tb = cKDTree(b)

    da = tb.query(a, k=1)[0].mean()
    db = ta.query(b, k=1)[0].mean()

    return 0.5 * (float(da) + float(db))


def select_train_medoid(train_worms, mean, std):
    xyzs = [
        normalized_xyz(w, mean, std)
        for w in train_worms
    ]

    sums = np.zeros(
        len(train_worms),
        dtype=np.float64,
    )

    for i, j in combinations(
        range(len(train_worms)), 2
    ):
        d = symmetric_chamfer(
            xyzs[i], xyzs[j]
        )
        sums[i] += d
        sums[j] += d

    denom = max(
        len(train_worms) - 1, 1
    )
    average = sums / denom

    index = int(np.argmin(average))

    return index, average


def construct_graph_edges(xyz):
    """
    Official NGM-v2 uses triangulated keypoint graphs.

    Atanas is 3-D, so:
      1. use 3-D Delaunay tetrahedralization;
      2. connect vertices sharing a simplex;
      3. make edges symmetric.

    SplineConv in official NGM-v2 expects 2-D pseudo coordinates.
    We encode a 3-D edge direction by spherical:
      azimuth, elevation
    both normalized into [0,1].
    """
    n = len(xyz)

    undirected = set()

    if n >= 4:
        try:
            tri = Delaunay(
                xyz,
                qhull_options="QJ",
            )

            for simplex in tri.simplices:
                for a, b in combinations(
                    map(int, simplex), 2
                ):
                    if a != b:
                        undirected.add(
                            (
                                min(a, b),
                                max(a, b),
                            )
                        )

        except Exception:
            undirected.clear()

    # Robust fallback only if Delaunay fails.
    if not undirected:
        k = min(6, max(1, n - 1))

        tree = cKDTree(xyz)
        _, neighbors = tree.query(
            xyz,
            k=k + 1,
        )

        for i in range(n):
            for j in np.asarray(
                neighbors[i]
            ).reshape(-1)[1:]:

                j = int(j)

                if i != j:
                    undirected.add(
                        (
                            min(i, j),
                            max(i, j),
                        )
                    )

    src = []
    dst = []

    for a, b in sorted(undirected):
        src += [a, b]
        dst += [b, a]

    edge_index = np.asarray(
        [src, dst],
        dtype=np.int64,
    )

    delta = (
        xyz[edge_index[1]]
        - xyz[edge_index[0]]
    )

    xy_norm = np.sqrt(
        np.maximum(
            delta[:, 0] ** 2
            + delta[:, 1] ** 2,
            1e-12,
        )
    )

    azimuth = (
        np.arctan2(
            delta[:, 1],
            delta[:, 0],
        )
        + np.pi
    ) / (2.0 * np.pi)

    elevation = (
        np.arctan2(
            delta[:, 2],
            xy_norm,
        )
        + np.pi / 2.0
    ) / np.pi

    pseudo = np.stack(
        [azimuth, elevation],
        axis=1,
    ).astype(np.float32)

    pseudo = np.clip(
        pseudo,
        0.0,
        1.0,
    )

    return edge_index, pseudo


class AtanasNGMv2(nn.Module):
    """
    ThinkMatch NGM-v2 official core, adapted to neuron graphs.

    Replaced:
        image/VGG feature extractor
    with:
        XYZ -> MLP

    Preserved:
        Siamese SplineConv
        node-to-edge feature construction
        weighted unary/quadratic affinity
        association graph
        NGM graph neural layers
        Sinkhorn
    """

    def __init__(
        self,
        feature_dim=64,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.node_projection = nn.Sequential(
            nn.Linear(
                3,
                feature_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                feature_dim,
                feature_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(
                feature_dim
            ),
        )

        # OFFICIAL ThinkMatch NGM-v2
        self.spline_conv = (
            SiameseSConvOnNodes(
                input_node_dim=feature_dim
            )
        )

        # OFFICIAL ThinkMatch
        self.edge_builder = (
            SiameseNodeFeaturesToEdgeFeatures(
                total_num_nodes=feature_dim
            )
        )

        # OFFICIAL affinity module
        self.vertex_affinity = (
            InnerProductWithWeightsAffinity(
                2 * feature_dim,
                feature_dim,
            )
        )

        self.edge_affinity = (
            InnerProductWithWeightsAffinity(
                2 * feature_dim,
                feature_dim,
            )
        )

        # Same NGM configuration:
        # GNN_FEAT = [16,16,16]
        # SK_EMB = 1
        gnn_features = [
            16, 16, 16
        ]

        self.gnn_layers = nn.ModuleList()

        for layer_id, out_dim in enumerate(
            gnn_features
        ):
            if layer_id == 0:
                in_dim = 1
            else:
                in_dim = (
                    gnn_features[
                        layer_id - 1
                    ]
                    + 1
                )

            # OFFICIAL sparse NGM GNN
            layer = PYGNNLayer(
                in_node_features=in_dim,
                in_edge_features=1,
                out_node_features=(
                    out_dim + 1
                ),
                out_edge_features=out_dim,
                sk_channel=1,
                sk_tau=0.05,
                edge_emb=False,
            )

            self.gnn_layers.append(
                layer
            )

        self.classifier = nn.Linear(
            gnn_features[-1] + 1,
            1,
        )

        # OFFICIAL Sinkhorn settings
        self.sinkhorn = Sinkhorn(
            max_iter=20,
            tau=0.05,
            epsilon=1e-10,
        )

    def encode_graph(
        self,
        xyz,
        edge_index,
        pseudo,
    ):
        x = self.node_projection(
            xyz
        )

        # Official NGM-v2 normalizes CNN features over channels.
        x = torch.nn.functional.normalize(
            x,
            p=2,
            dim=-1,
        )

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=pseudo,
        )

        graph = Batch.from_data_list(
            [graph]
        )

        graph = self.spline_conv(
            graph
        )

        graph = (
            self.edge_builder(
                graph
            )[0]
        )

        # Geometry analogue of
        # original global CNN feature.
        global_feature = (
            graph.x.mean(dim=0)
        )

        return (
            graph,
            global_feature,
        )

    def forward(
        self,
        graph1,
        graph2,
    ):
        xyz1, edges1, pseudo1 = graph1
        xyz2, edges2, pseudo2 = graph2

        g1, global1 = self.encode_graph(
            xyz1,
            edges1,
            pseudo1,
        )

        g2, global2 = self.encode_graph(
            xyz2,
            edges2,
            pseudo2,
        )

        global_weights = torch.cat(
            [
                global1,
                global2,
            ],
            dim=-1,
        )

        # Official NGM-v2 normalizes global weights before affinity.
        global_weights = torch.nn.functional.normalize(
            global_weights,
            p=2,
            dim=0,
        )

        # Unary affinity Kp
        Kp = self.vertex_affinity(
            [g1.x],
            [g2.x],
            [global_weights],
        )[0]

        # Quadratic/edge affinity Ke
        Ke = 0.5 * self.edge_affinity(
            [g1.edge_attr],
            [g2.edge_attr],
            [global_weights],
        )[0]

        n1 = int(
            g1.x.shape[0]
        )
        n2 = int(
            g2.x.shape[0]
        )

        e1_src = g1.edge_index[0]
        e1_dst = g1.edge_index[1]

        e2_src = g2.edge_index[0]
        e2_dst = g2.edge_index[1]

        E1 = int(
            e1_src.numel()
        )
        E2 = int(
            e2_src.numel()
        )

        # Association graph node ordering follows
        # official NGM:
        # pair(i,j) -> j*n1 + i
        assoc_src = (
            e2_src.repeat(E1)
            * n1
            + e1_src.repeat_interleave(
                E2
            )
        )

        assoc_dst = (
            e2_dst.repeat(E1)
            * n1
            + e1_dst.repeat_interleave(
                E2
            )
        )

        # -------------------------------------------------
        # Sparse Lawler affinity matrix.
        #
        # This is mathematically the same construction used
        # by ThinkMatch construct_sparse_aff_mat:
        #
        #   - Ke gives off-diagonal association edges
        #   - Kp gives first-order unary terms on the diagonal
        #
        # We construct it directly with torch_sparse to avoid
        # ThinkMatch's legacy sparse_dot CUDA extension.
        # Association-node convention:
        #
        #       pair(i,j) -> j * n1 + i
        # -------------------------------------------------

        association_size = n1 * n2

        edge_rows = assoc_src.long()
        edge_cols = assoc_dst.long()
        edge_values = Ke.contiguous().reshape(-1)

        if edge_values.numel() != edge_rows.numel():
            raise RuntimeError(
                f"Ke/index mismatch: values={edge_values.numel()} "
                f"edges={edge_rows.numel()}"
            )

        # Kp is [n1,n2].
        # Transpose before flattening because association node
        # id is j*n1+i.
        diag_values = (
            Kp.transpose(0, 1)
            .contiguous()
            .reshape(-1)
        )

        if diag_values.numel() != association_size:
            raise RuntimeError(
                f"Kp size mismatch: {diag_values.numel()} "
                f"vs {association_size}"
            )

        diag_idx = torch.arange(
            association_size,
            device=xyz1.device,
            dtype=torch.long,
        )

        row_idx = torch.cat(
            [edge_rows, diag_idx],
            dim=0,
        )

        col_idx = torch.cat(
            [edge_cols, diag_idx],
            dim=0,
        )

        K_value = torch.cat(
            [edge_values, diag_values],
            dim=0,
        )

        association_graph = SparseTensor(
            row=row_idx,
            col=col_idx,
            value=K_value,
            sparse_sizes=(
                association_size,
                association_size,
            ),
        ).coalesce()

        # FIRST_ORDER=True:
        # initial association-node embedding
        # is unary affinity.
        embedding = (
            Kp.transpose(0, 1)
            .contiguous()
            .view(
                1,
                association_size,
                1,
            )
        )

        n1_tensor = torch.tensor(
            [n1],
            device=xyz1.device,
            dtype=torch.long,
        )

        n2_tensor = torch.tensor(
            [n2],
            device=xyz1.device,
            dtype=torch.long,
        )

        for layer in self.gnn_layers:
            embedding = layer(
                association_graph,
                embedding,
                n1_tensor,
                n2_tensor,
                idx=0,
            )

        logits = self.classifier(
            embedding
        )

        # Exact official reshape convention
        score = (
            logits
            .view(
                1,
                n2,
                n1,
            )
            .transpose(1, 2)
        )

        ds = self.sinkhorn(
            score,
            n1_tensor,
            n2_tensor,
            dummy_row=True,
        )

        return ds


def make_graph(
    xyz,
    device,
):
    edge_index, pseudo = (
        construct_graph_edges(
            xyz
        )
    )

    return (
        torch.as_tensor(
            xyz,
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            edge_index,
            dtype=torch.long,
            device=device,
        ),
        torch.as_tensor(
            pseudo,
            dtype=torch.float32,
            device=device,
        ),
    )


def training_intersection(
    worm1,
    worm2,
    mean,
    std,
    min_common,
):
    map1 = unique_identity_map(
        worm1
    )
    map2 = unique_identity_map(
        worm2
    )

    common = sorted(
        set(map1)
        & set(map2)
    )

    if len(common) < min_common:
        return None

    idx1 = np.asarray(
        [
            map1[label]
            for label in common
        ],
        dtype=np.int64,
    )

    idx2 = np.asarray(
        [
            map2[label]
            for label in common
        ],
        dtype=np.int64,
    )

    xyz1 = normalized_xyz(
        worm1,
        mean,
        std,
    )[idx1]

    xyz2 = normalized_xyz(
        worm2,
        mean,
        std,
    )[idx2]

    # Both are ordered by the same
    # sorted identity list.
    gt = np.eye(
        len(common),
        dtype=np.float32,
    )

    return (
        xyz1,
        xyz2,
        gt,
    )


@torch.no_grad()
def evaluate_split(
    model,
    worms,
    template,
    mean,
    std,
    device,
):
    model.eval()

    template_map = (
        unique_identity_map(
            template
        )
    )

    candidate_labels = list(
        template_map.keys()
    )

    candidate_indices = np.asarray(
        [
            template_map[x]
            for x in candidate_labels
        ],
        dtype=np.int64,
    )

    candidate_lookup = {
        label: i
        for i, label in enumerate(
            candidate_labels
        )
    }

    template_graph = make_graph(
        normalized_xyz(
            template,
            mean,
            std,
        ),
        device,
    )

    total_queries = 0
    top1_correct = 0
    top5_correct = 0
    reciprocal_rank = 0.0
    hungarian_correct = 0

    possible_queries = 0
    covered_queries = 0

    per_worm = []

    for worm in worms:
        source_map = (
            unique_identity_map(
                worm
            )
        )

        possible_queries += len(
            source_map
        )

        labels = [
            label
            for label in source_map
            if label in template_map
        ]

        covered_queries += len(
            labels
        )

        if not labels:
            continue

        source_indices = np.asarray(
            [
                source_map[label]
                for label in labels
            ],
            dtype=np.int64,
        )

        source_graph = make_graph(
            normalized_xyz(
                worm,
                mean,
                std,
            ),
            device,
        )

        ds = model(
            source_graph,
            template_graph,
        )[0]

        ds = (
            ds.detach()
            .float()
            .cpu()
            .numpy()
        )

        score_matrix = ds[
            np.ix_(
                source_indices,
                candidate_indices,
            )
        ]

        gt_columns = np.asarray(
            [
                candidate_lookup[
                    label
                ]
                for label in labels
            ],
            dtype=np.int64,
        )

        worm_top1 = 0

        for row, gt_col in enumerate(
            gt_columns
        ):
            target = score_matrix[
                row,
                gt_col,
            ]

            rank = (
                1
                + int(
                    np.sum(
                        score_matrix[row]
                        > target
                    )
                )
            )

            is_top1 = int(
                rank == 1
            )

            worm_top1 += is_top1
            top1_correct += is_top1

            top5_correct += int(
                rank
                <= min(
                    5,
                    score_matrix.shape[1],
                )
            )

            reciprocal_rank += (
                1.0 / rank
            )

        row_idx, col_idx = (
            linear_sum_assignment(
                -score_matrix
            )
        )

        assignment = {
            int(r): int(c)
            for r, c
            in zip(
                row_idx,
                col_idx,
            )
        }

        worm_hungarian = sum(
            int(
                assignment.get(
                    row,
                    -1,
                )
                == int(gt_col)
            )
            for row, gt_col
            in enumerate(gt_columns)
        )

        hungarian_correct += (
            worm_hungarian
        )

        n_queries = len(labels)
        total_queries += n_queries

        per_worm.append(
            {
                "uid": worm.uid,
                "queries": n_queries,
                "top1": (
                    worm_top1
                    / n_queries
                ),
                "hungarian": (
                    worm_hungarian
                    / n_queries
                ),
            }
        )

    q = max(
        total_queries,
        1,
    )

    return {
        "queries": total_queries,
        "top1": (
            top1_correct / q
        ),
        "top5": (
            top5_correct / q
        ),
        "mrr": (
            reciprocal_rank / q
        ),
        "hungarian": (
            hungarian_correct / q
        ),
        "coverage": (
            covered_queries
            / max(
                possible_queries,
                1,
            )
        ),
        "possible_unique_queries":
            possible_queries,
        "covered_unique_queries":
            covered_queries,
        "per_worm": per_worm,
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/home/ubuntu/klb/nuclr/nuclr/"
            "Data/Atanas_SF_unified_000776/"
            "cv5_grouped_v1"
        ),
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--pairs-per-epoch",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--min-common",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3,
    )

    parser.add_argument(
        "--max-val-worms",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        args.device
    )

    fold_root = (
        args.data_root
        / f"fold_{args.fold}"
    )

    train_paths = sorted(
        (
            fold_root / "train"
        ).glob("*.npz")
    )

    val_paths = sorted(
        (
            fold_root / "val"
        ).glob("*.npz")
    )

    if not train_paths:
        raise RuntimeError(
            "No train files"
        )

    if not val_paths:
        raise RuntimeError(
            "No val files"
        )

    # IMPORTANT:
    # test directory is intentionally
    # NOT loaded here.

    train = [
        load_worm(p)
        for p in train_paths
    ]

    val = [
        load_worm(p)
        for p in val_paths
    ]

    mean, std = fit_train_scaler(
        train
    )

    medoid_index, medoid_scores = (
        select_train_medoid(
            train,
            mean,
            std,
        )
    )

    template = train[
        medoid_index
    ]

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        commit = (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            )
            .strip()
        )
    except Exception:
        commit = "UNKNOWN"

    protocol = {
        "method":
            "NGM-v2 official core adapted",
        "input":
            "geometry-only",
        "normalization":
            "train-only per-axis z-score",
        "graph":
            "3D Delaunay, symmetric",
        "training_filter":
            "intersection of unique supervised IDs",
        "evaluation":
            "full source/template graphs; "
            "strict unique identities",
        "template_selection":
            "train-only symmetric Chamfer medoid",
        "checkpoint_selection":
            "val Hungarian, then Top1",
        "fold": args.fold,
        "seed": args.seed,
        "template":
            template.path.name,
        "template_medoid_chamfer":
            float(
                medoid_scores[
                    medoid_index
                ]
            ),
        "train_mean":
            mean.tolist(),
        "train_std":
            std.tolist(),
        "thinkmatch_commit":
            commit,
    }

    (
        args.out
        / "protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
        )
        + "\n"
    )

    usable_pairs = []

    for i, j in combinations(
        range(len(train)),
        2,
    ):
        item = training_intersection(
            train[i],
            train[j],
            mean,
            std,
            args.min_common,
        )

        if item is not None:
            usable_pairs.append(
                (i, j)
            )

    if not usable_pairs:
        raise RuntimeError(
            "No usable train pairs"
        )

    print(
        f"[DATA] train={len(train)} "
        f"val={len(val)} "
        f"usable_pairs={len(usable_pairs)}"
    )

    print(
        f"[TEMPLATE] "
        f"{template.path.name} "
        f"medoid_chamfer="
        f"{medoid_scores[medoid_index]:.6f}"
    )

    model = AtanasNGMv2(
        feature_dim=args.feature_dim
    ).to(device)

    criterion = PermutationLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .MultiStepLR(
            optimizer,
            milestones=[
                2, 4, 6, 8, 10
            ],
            gamma=0.5,
        )
    )

    best_score = (
        -1.0,
        -1.0,
    )

    best_path = (
        args.out
        / "best.pt"
    )

    if args.max_val_worms > 0:
        val_selection = val[
            : args.max_val_worms
        ]
    else:
        val_selection = val

    for epoch in range(
        args.epochs
    ):
        model.train()

        order = list(
            usable_pairs
        )

        random.shuffle(
            order
        )

        if args.pairs_per_epoch > 0:
            if len(order) >= (
                args.pairs_per_epoch
            ):
                order = order[
                    : args.pairs_per_epoch
                ]
            else:
                order = [
                    random.choice(
                        usable_pairs
                    )
                    for _ in range(
                        args.pairs_per_epoch
                    )
                ]

        losses = []

        for i, j in order:
            item = training_intersection(
                train[i],
                train[j],
                mean,
                std,
                args.min_common,
            )

            if item is None:
                continue

            xyz1, xyz2, gt = item

            graph1 = make_graph(
                xyz1,
                device,
            )

            graph2 = make_graph(
                xyz2,
                device,
            )

            gt_tensor = (
                torch.as_tensor(
                    gt,
                    dtype=torch.float32,
                    device=device,
                )
                .unsqueeze(0)
            )

            ns = torch.tensor(
                [len(xyz1)],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            ds = model(
                graph1,
                graph2,
            )

            loss = criterion(
                ds,
                gt_tensor,
                ns,
                ns,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.detach()
                )
            )

        val_metrics = evaluate_split(
            model,
            val_selection,
            template,
            mean,
            std,
            device,
        )

        score = (
            val_metrics[
                "hungarian"
            ],
            val_metrics[
                "top1"
            ],
        )

        print(
            f"[E{epoch+1:02d}] "
            f"loss={np.mean(losses):.5f} "
            f"VAL Q="
            f"{val_metrics['queries']} "
            f"Top1="
            f"{100*val_metrics['top1']:.2f}% "
            f"Top5="
            f"{100*val_metrics['top5']:.2f}% "
            f"MRR="
            f"{val_metrics['mrr']:.4f} "
            f"Hung="
            f"{100*val_metrics['hungarian']:.2f}% "
            f"Cov="
            f"{100*val_metrics['coverage']:.2f}%"
        )

        if score > best_score:
            best_score = score

            torch.save(
                {
                    "model":
                        model.state_dict(),
                    "epoch":
                        epoch + 1,
                    "val":
                        val_metrics,
                    "mean":
                        mean,
                    "std":
                        std,
                    "template":
                        template.path.name,
                    "fold":
                        args.fold,
                    "seed":
                        args.seed,
                    "feature_dim":
                        args.feature_dim,
                },
                best_path,
            )

        scheduler.step()

    payload = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        payload["model"]
    )

    full_val_metrics = (
        evaluate_split(
            model,
            val,
            template,
            mean,
            std,
            device,
        )
    )

    (
        args.out
        / "val_metrics.json"
    ).write_text(
        json.dumps(
            full_val_metrics,
            indent=2,
        )
        + "\n"
    )

    print(
        f"[BEST] epoch="
        f"{payload['epoch']} "
        f"VAL Top1="
        f"{100*full_val_metrics['top1']:.2f}% "
        f"Hung="
        f"{100*full_val_metrics['hungarian']:.2f}%"
    )

    if args.skip_test:
        print(
            "[LOCK] --skip-test: "
            "TEST WAS NOT READ"
        )
        return

    lock_payload = {
        "status":
            "LOCKED_BEFORE_TEST",
        "best_epoch":
            payload["epoch"],
        "selection_metric":
            "val Hungarian then Top1",
        "template":
            template.path.name,
        "fold":
            args.fold,
        "seed":
            args.seed,
    }

    (
        args.out
        / "LOCKED_BEFORE_TEST.json"
    ).write_text(
        json.dumps(
            lock_payload,
            indent=2,
        )
        + "\n"
    )

    print(
        "[LOCKED_BEFORE_TEST]"
    )

    # Only now is TEST read.
    test_paths = sorted(
        (
            fold_root / "test"
        ).glob("*.npz")
    )

    if not test_paths:
        raise RuntimeError(
            "No test files"
        )

    test = [
        load_worm(p)
        for p in test_paths
    ]

    test_metrics = evaluate_split(
        model,
        test,
        template,
        mean,
        std,
        device,
    )

    (
        args.out
        / "test_metrics.json"
    ).write_text(
        json.dumps(
            test_metrics,
            indent=2,
        )
        + "\n"
    )

    print("=" * 76)
    print(
        f"NGM-v2 — ATANAS "
        f"FOLD{args.fold} "
        f"SEED{args.seed} "
        f"— LOCKED TEST"
    )
    print("=" * 76)

    print(
        f"Queries     : "
        f"{test_metrics['queries']}"
    )

    print(
        f"Top-1       : "
        f"{100*test_metrics['top1']:.2f}%"
    )

    print(
        f"Top-5       : "
        f"{100*test_metrics['top5']:.2f}%"
    )

    print(
        f"MRR         : "
        f"{test_metrics['mrr']:.4f}"
    )

    print(
        f"Hungarian   : "
        f"{100*test_metrics['hungarian']:.2f}%"
    )

    print(
        f"Coverage    : "
        f"{100*test_metrics['coverage']:.2f}%"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
