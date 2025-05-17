# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_t2v_14B import t2v_14B

# the config of t2i_14B is the same as t2v_14B
t2i_14B = copy.deepcopy(t2v_14B)
t2i_14B.__name__ = 'Config: Wan T2I 14B'

WAN_CONFIGS = {
    't2v-14B': t2v_14B,
    't2i-14B': t2i_14B,
}

SIZE_CONFIGS = {
    "1920*1056": (1920, 1056),
    "1920*1072": (1920, 1072),
    "1920*832": (1920, 832),
    "1280*560": (1280, 560),
    "560*1280": (560, 1280),
    "1056*1920": (1056, 1920),
    "832*1920": (832, 1920),
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
}

SUPPORTED_SIZES = {
    't2v-14B': ('720*1280', '1280*720', '480*832', '832*480', "1920*1056", "1056*1920", "1920*832", "832*1920", "1920*1072", "1072*1920", "1280*560", "560*1280"),
    't2i-14B': tuple(SIZE_CONFIGS.keys()),
}
