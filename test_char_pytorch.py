import torch
import torch.nn as nn
from torch.optim import SGD

from lid.charEmbBag import CharacterLID
from data.europarl import EUROPARL, europarl_collate_fn

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

europarl_basedir = "/home/david/Programming/data/WMT/europarl"

ds = EUROPARL(europarl_basedir, split="test", max_samples=100, sample_len=5, truncate_to=5000)
dl = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True, collate_fn=europarl_collate_fn)

print("ds len: {}".format(len(ds)))

m = CharacterLID(ds.charset_size + 1, num_emb_out=100, num_classes=21).to(device)
m.eval()
predictions = torch.LongTensor()
for i, (mb, labels, ns) in enumerate(dl):
    mb, labels = mb.to(device), labels.to(device)
    out = m(mb)
    for ns_i in ns:
        out_i = out[:ns_i]
        out_i = out_i.sum(dim=0)
        preds_i = torch.topk(out_i, 1)[1]
        predictions = torch.cat((predictions, preds_i))
print(predictions.size())
