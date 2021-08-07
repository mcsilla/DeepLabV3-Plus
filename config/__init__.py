#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""

import config.camvid_resnet50
import config.human_parsing_resnet50
import config.articles_tpu
import config.articles_tpu_test
import config.articles_gpu_test
import config.article_segmentation

CONFIG_MAP = {
    'camvid_resnet50': config.camvid_resnet50.CONFIG,
    'human_parsing_resnet50': config.human_parsing_resnet50.CONFIG,
    'articles_tpu': config.articles_tpu.CONFIG,
    'article_segmentation': config.article_segmentation.CONFIG,
    'articles_tpu_test': config.articles_tpu_test.CONFIG,
    'articles_gpu_test': config.articles_gpu_test.CONFIG
}
