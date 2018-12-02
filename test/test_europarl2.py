import torch
from data.europarl import EUROPARL2, europarl_collate_fn

europarl_basedir = "/home/david/Programming/data/WMT/europarl"

ds = EUROPARL2(europarl_basedir, split="dev_train", max_samples=100, sample_len=5, truncate_to=5000)
dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=europarl_collate_fn)
epochs = 2
for epoch in range(epochs):
    for x in dl:
        x
