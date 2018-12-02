import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import SGD

from data.europarl import EUROPARLFT
from lid.ftModel import FTModel

from fastai import *

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

europarl_basedir = "/home/david/Programming/data/WMT/europarl"

batch_size_training = 1000

ds = EUROPARLFT(europarl_basedir, split="split_train")
dl = data.DataLoader(ds, batch_size=batch_size_training, shuffle=False)
ds_valid = EUROPARLFT(europarl_basedir, split="split_valid")
dl_valid = data.DataLoader(ds_valid, batch_size=100, shuffle=False)
ds_test = EUROPARLFT(europarl_basedir, split="test")
dl_test = data.DataLoader(ds_test, batch_size=20, shuffle=False)
db = DataBunch(dl, dl_valid, test_dl=dl_test)

emb_dim = ds.embbag.weight.data.size(1)

m = FTModel(input_dim = emb_dim, layer_sizes = [1000,200])

criterion = nn.CrossEntropyLoss()

learner = Learner(db, m, loss_func = criterion, metrics=accuracy)
learner.unfreeze()

learner.fit_one_cycle(10,1e-2)

learner.save("model_1000_200_layernorm_epoch5")
