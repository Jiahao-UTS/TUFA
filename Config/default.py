#--------------------------------------
#author: Jiahao.Xia
#--------------------------------------

from yacs.config import CfgNode as CN

import os

_C = CN()
_C.GPUS = (0, )
_C.WORKERS = 0
_C.PIN_MEMORY = True

_C.DATASET = CN()
_C.DATASET.ROOT = "./Dataloader"
_C.DATASET.CHANNEL = 3
_C.DATASET.DATASET = 'ALL'

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1

_C.MODEL = CN()
_C.MODEL.NAME = "Prompt_Face"
_C.MODEL.IMG_SIZE = 256
_C.MODEL.TYPE = 'base'

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.HYPERPARAMETERS = CN()

_C.ALL = CN()
_C.ALL.ROOT = './Data/'
_C.ALL.FRACTION = 1.20
_C.ALL.DATA_FORMAT = "RGB"



