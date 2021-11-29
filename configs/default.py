from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()


# ----- BASIC SETTINGS -----
_C.NAME = "default"
_C.REDIRECTOR = True

_C.RELOAD = CN()
_C.RELOAD.TYPE = False
_C.RELOAD.NAME = 'backbone'
_C.RELOAD.PTH = ''


_C.DISTRIBUTTED = False

# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.TYPE = "pedes"
_C.DATASET.NAME = "PA100k"
_C.DATASET.TARGETTRANSFORM = []
_C.DATASET.ZERO_SHOT = False
_C.DATASET.LABEL = 'eval'  # train on all labels, test on part labels (35 for peta, 51 for rap)
_C.DATASET.TRAIN_SPLIT = 'trainval'
_C.DATASET.VAL_SPLIT = 'val'
_C.DATASET.TEST_SPLIT = 'test'
_C.DATASET.HEIGHT = 256
_C.DATASET.WIDTH = 192

# ----- BACKBONE BUILDER -----
_C.BACKBONE = CN()
_C.BACKBONE.TYPE = "resnet50"
_C.BACKBONE.MULTISCALE = False

# ----- MODULE BUILDER -----
# _C.MODULE = CN()
# _C.MODULE.TYPE = "GAP"

# ----- CLASSIFIER BUILDER -----
_C.CLASSIFIER = CN()
_C.CLASSIFIER.TYPE = "base"
_C.CLASSIFIER.NAME = ""
_C.CLASSIFIER.POOLING = "avg"
_C.CLASSIFIER.BN = False
_C.CLASSIFIER.SCALE = 1

# ----- TRAIN BUILDER -----
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.MAX_EPOCH = 30
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.CLIP_GRAD = False
_C.TRAIN.BN_WD = True

_C.TRAIN.DATAAUG = CN()
_C.TRAIN.DATAAUG.TYPE = 'base'
_C.TRAIN.DATAAUG.AUTOAUG_PROB = 0.5

_C.TRAIN.EMA = CN()
_C.TRAIN.EMA.ENABLE = False
_C.TRAIN.EMA.DECAY = 0.9998
_C.TRAIN.EMA.FORCE_CPU = False

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = "plateau"
_C.TRAIN.LR_SCHEDULER.LR_STEP = [0,]
_C.TRAIN.LR_SCHEDULER.LR_FT = 1e-2
_C.TRAIN.LR_SCHEDULER.LR_NEW = 1e-2
_C.TRAIN.LR_SCHEDULER.WMUP_COEF = 0.1


_C.TRAIN.AUX_LOSS_START = -1

# ----- INFER BUILDER -----

_C.INFER = CN()
_C.INFER.SAMPLING = False

# ----- LOSS BUILDER -----
_C.LOSS = CN()
_C.LOSS.TYPE = "bce"
_C.LOSS.SAMPLE_WEIGHT = ""  # None
_C.LOSS.LOSS_WEIGHT = [1, ]
_C.LOSS.SIZESUM = True   # for a sample, BCE losses is the summation of all label instead of the average.

_C.METRIC = CN()
_C.METRIC.TYPE = 'pedestrian'

# ------ visualization ---------
_C.VIS = CN()
_C.VIS.CAM = 'valid'
_C.VIS.TENSORBOARD = CN()
_C.VIS.TENSORBOARD.ENABLE = True

_C.VIS.VISDOM = False


# ----------- Transformer -------------
_C.TRANS = CN()
_C.TRANS.DIM_HIDDEN = 256
_C.TRANS.DROPOUT = 0.1
_C.TRANS.NHEADS = 8
_C.TRANS.DIM_FFD = 2048
_C.TRANS.ENC_LAYERS = 6
_C.TRANS.DEC_LAYERS = 6
_C.TRANS.PRE_NORM = False
_C.TRANS.EOS_COEF = 0.1
_C.TRANS.NUM_QUERIES = 100


# testing
# _C.TEST = CN()
# _C.TEST.BATCH_SIZE = 32
# _C.TEST.NUM_WORKERS = 8
# _C.TEST.MODEL_FILE = ""
#
# _C.TRANSFORMS = CN()
# _C.TRANSFORMS.TRAIN_TRANSFORMS = ("random_resized_crop", "random_horizontal_flip")
# _C.TRANSFORMS.TEST_TRANSFORMS = ("shorter_resize_for_crop", "center_crop")
#
# _C.TRANSFORMS.PROCESS_DETAIL = CN()
# _C.TRANSFORMS.PROCESS_DETAIL.RANDOM_CROP = CN()
# _C.TRANSFORMS.PROCESS_DETAIL.RANDOM_CROP.PADDING = 4
# _C.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP = CN()
# _C.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.SCALE = (0.08, 1.0)
# _C.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.RATIO = (0.75, 1.333333333)


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)  # update cfg
    # cfg.merge_from_list(args.opts)

    cfg.freeze()
