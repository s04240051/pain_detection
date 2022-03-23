
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yacs.config import CfgNode as CN
_C = CN()

_C.KEYPOINT_FILE = ""
_C.TRAIN_TEST_SPLIT = ""
_C.CROP_IMAGE = ""
_C.FLOW_IMAGE = ""
_C.OUT_DIR = r"out"
_C.TRAIN_INITIAL_WEIGHT = ""
_C.TEST_INITIAL_WEIGHT = ""
_C.AUXILIARY_FILE = r"D:\pose\pain\data\pain_data\annotation\split\second_label.json"
_C.FORMAT_DATASET = True
_C.DATASET_PATH = r"D:\pose\pain\data\pain_data"
_C.CHECKPOINTS_FOLD = "checkpoints"
_C.TRAIN_RECORD = ""
_C.TEST_RECORD = ""
_C.VAL_RECORD = ""
_C.NUM_GPUS = 1
_C.RNG_SEED = 1
_C.ENABLE_TRAIN = False
_C.ENABLE_VAL = True
_C.ENABLE_TEST = False
_C.SAVE_PREDS = True

_C.MODEL = CN()
_C.MODEL.MODEL_NAME = "Two_stream_model"
_C.MODEL.TYPE = "two_stream" #["two_stream", "rgb", "kp", "flow"]
_C.MODEL.FUSION_METHOD = "Late" #["Late", "Bilinear", "Concat"]
_C.MODEL.BILINEAR_OUT_DIM = 512
_C.MODEL.ATTENTION = True
_C.MODEL.NUM_CLSTM_LAYERS = 4
_C.MODEL.CLSTM_HIDDEN_SIZE = 32
_C.MODEL.LSTM_HIDDEN_SIZE = 32
_C.MODEL.NUM_LSTM_LAYERS = 4
_C.MODEL.LSTM_INPUT_SIZE = 34
_C.MODEL.IMG_SIZE = (224, 224)
_C.MODEL.NUM_LABELS = 2

_C.DATA = CN()
_C.DATA.REQUIRE_AUX = True
_C.DATA.DATA_TYPE = "aux" #["simple","diff", "aux"]
_C.DATA.EXTRA_LABEL = False
_C.DATA.AUG = True
_C.DATA.CLIP_LENGTH = 8
_C.DATA.BALANCE_POLICY = 0 # 0: no balance, 1: add weight to loss, 2: add weight to sampler, 
_C.DATA.MEAN = [0.45, 0.45, 0.45]
_C.DATA.STD = [0.225, 0.225, 0.225]
_C.DATA.MEAN_FLOW = []
_C.DATA.STD_FLOW = []
_C.DATA.CROP_THRESHOLD = [300,  450, 600]

_C.SOLVER = CN()
_C.SOLVER.METHOD = "sgd"
_C.SOLVER.LR_POLICY = "steps_with_relative_lrs" # consine or steps_with_relative_lrs
_C.SOLVER.BASE_LR = 0.1
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.DAMPENING = 0.0
_C.SOLVER.NESTEROV = True
_C.SOLVER.WARMUP_EPOCHS = 20 # 0.0 if steps_with_relative_lrs
_C.SOLVER.WARMUP_START_LR = 0.01
_C.SOLVER.COSINE_AFTER_WARMUP = False
_C.SOLVER.COSINE_END_LR = 0.0
_C.SOLVER.LRS = [1, 0.1, 0.01, 0.001]
_C.SOLVER.STEPS = [0,44,88,118]
_C.SOLVER.MAX_EPOCH = 118

_C.RUN = CN()
_C.RUN.TEST_BATCH_SIZE = 1
_C.RUN.TRAIN_BATCH_SIZE = 16
_C.RUN.NUM_WORKS = 2
_C.RUN.AUTO_RESUME = True
_C.RUN.SAVE_STEP = 3
_C.RUN.SAVE_LAST = 5

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()