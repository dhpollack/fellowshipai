import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import SGD

from data.europarl import EUROPARLFT
from lid.ftModel import FTModel

from fastai import Learner, DataBunch

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

europarl_basedir = "/home/david/Programming/data/WMT/europarl"

batch_size_training = 1000

ds = EUROPARLFT(europarl_basedir, split="train")
dl = data.DataLoader(ds, batch_size=batch_size_training, shuffle=False)
ds_valid = EUROPARLFT(europarl_basedir, split="dev_train")
dl_valid = data.DataLoader(ds_valid, batch_size=20, shuffle=False)
db = DataBunch(dl, dl_valid)

emb_dim = ds.embbag.weight.data.size(1)

m = FTModel(input_dim = emb_dim, layer_sizes = [1000,200])

learner = Learner(db, m)
learner.load("model_1000_200_layernorm_epoch5")

ds_test = EUROPARLFT(europarl_basedir, split="test")
dl_test = data.DataLoader(ds_test, batch_size=20, shuffle=False)

acc = torch.tensor(0)

for mb, labels in dl_test:
    out = learner.model(mb)
    pred = torch.topk(out, 1)[1].reshape(-1)
    acc += (pred == labels).sum()
print(acc.item() / len(ds_test))
