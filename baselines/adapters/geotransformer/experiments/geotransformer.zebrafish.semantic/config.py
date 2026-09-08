import os
import os.path as osp

from easydict import EasyDict as edict
from geotransformer.utils.common import ensure_dir


_C = edict()

_C.seed = int(os.environ.get("SEED", "42"))

_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)

run_tag = os.environ.get("RUN_TAG", f"seed{_C.seed}")

_C.output_dir = osp.join(
    _C.root_dir,
    "output",
    _C.exp_name,
    run_tag,
)
_C.snapshot_dir = osp.join(_C.output_dir, "snapshots")
_C.log_dir = osp.join(_C.output_dir, "logs")
_C.event_dir = osp.join(_C.output_dir, "events")

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
_C.data = edict()

_C.data.dataset_root = os.environ.get(
    "ZEBRAFISH_ROOT",
    "/home/ubuntu/klb/nuclr/nuclr/Data/"
    "Zebrafish_MPRT_LOFO8_60m/fold_1",
)

# minimum number of shared labelled neurons required for an animal pair
_C.data.min_shared = int(os.environ.get("MIN_SHARED", "20"))

# independently remove translation / global scale nuisance
_C.data.normalize = True


# ----------------------------------------------------------------------
# train / val
# ----------------------------------------------------------------------
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 4

_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 4


# ----------------------------------------------------------------------
# optimisation
# ----------------------------------------------------------------------
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.weight_decay = 1e-6

_C.optim.warmup_steps = int(os.environ.get("GT_WARMUP", "200"))
_C.optim.eta_init = 0.1
_C.optim.eta_min = 0.1

# smoke run can override this
_C.optim.max_iteration = int(os.environ.get("GT_MAX_ITERS", "20000"))
_C.optim.snapshot_steps = int(os.environ.get("GT_SNAPSHOT_STEPS", "1000"))
_C.optim.grad_acc_steps = 1


# ----------------------------------------------------------------------
# KPConv backbone
# points are normalized to roughly unit spatial scale
# ----------------------------------------------------------------------
_C.backbone = edict()

_C.backbone.num_stages = 3

_C.backbone.init_voxel_size = 0.05
_C.backbone.kernel_size = 15

_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0

_C.backbone.init_radius = (
    _C.backbone.base_radius * _C.backbone.init_voxel_size
)
_C.backbone.init_sigma = (
    _C.backbone.base_sigma * _C.backbone.init_voxel_size
)

_C.backbone.group_norm = 32

_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256


# ----------------------------------------------------------------------
# global matching
# ----------------------------------------------------------------------
_C.model = edict()

# neuron clouds are much smaller than ModelNet point clouds
_C.model.num_points_in_patch = int(
    os.environ.get("GT_PATCH_K", "32")
)

_C.model.num_sinkhorn_iterations = 50


# ----------------------------------------------------------------------
# coarse matching
# ----------------------------------------------------------------------
_C.coarse_matching = edict()

_C.coarse_matching.num_targets = 64
_C.coarse_matching.overlap_threshold = 0.0

_C.coarse_matching.num_correspondences = int(
    os.environ.get("GT_COARSE_K", "64")
)
_C.coarse_matching.dual_normalization = True


# ----------------------------------------------------------------------
# GeoTransformer
# keep the official geometric transformer architecture
# ----------------------------------------------------------------------
_C.geotransformer = edict()

_C.geotransformer.input_dim = 512
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256

_C.geotransformer.num_heads = 4

_C.geotransformer.blocks = [
    "self",
    "cross",
    "self",
    "cross",
    "self",
    "cross",
]

_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = "max"


# ----------------------------------------------------------------------
# coarse semantic loss
# ----------------------------------------------------------------------
_C.coarse_loss = edict()

_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24

_C.coarse_loss.positive_overlap = 0.0


# ----------------------------------------------------------------------
# overall loss
# ----------------------------------------------------------------------
_C.loss = edict()

_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


def make_cfg():
    return _C
