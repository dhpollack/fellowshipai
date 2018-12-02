import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import SGD

from lid.charEmbBag import CharacterLID
from data.europarl import EUROPARL2, europarl_fastai_collate_fn

from fastai import Learner, DataBunch

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

europarl_basedir = "/home/david/Programming/data/WMT/europarl"

ds = EUROPARL2(europarl_basedir, split="train", max_samples=100, sample_len=5, truncate_to=5000)
dl = data.DataLoader(ds, batch_size=2, shuffle=False)
db = DataBunch(dl, dl, collate_fn=europarl_fastai_collate_fn)

m = CharacterLID(ds.charset_size + 1, num_emb_out=100, num_classes=21).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = SGD(m.parameters(), lr=0.01, momentum=0.9, nesterov=True)

learner = Learner(db, m, loss_func = criterion)
learner.unfreeze()

learner.fit_one_cycle(1,1e-2)
