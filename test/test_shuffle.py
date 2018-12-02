from data.europarl import EUROPARL, europarl_collate_fn

import os

europarl_basedir = "/home/david/Programming/data/WMT/europarl"

ds = EUROPARL(europarl_basedir, split="train", max_samples=100, sample_len=5, truncate_to=5000)
