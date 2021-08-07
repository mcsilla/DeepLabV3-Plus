#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""


import config.articles_tpu

import config.page_segmentation

CONFIG_MAP = {
    'articles_tpu': config.articles_tpu.CONFIG,
    'page_segmentation': config.page_segmentation.CONFIG
}
