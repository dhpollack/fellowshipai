import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import SGD

from lid.charEmbBag import CharacterLID
from data.europarl import EUROPARL, europarl_collate_fn

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

europarl_basedir = "/home/david/Programming/data/WMT/europarl"

ds = EUROPARL(europarl_basedir, split="train", max_samples=100, sample_len=5, truncate_to=5000)
dl = data.DataLoader(ds, batch_size=2, shuffle=True, collate_fn=europarl_collate_fn)

m = CharacterLID(ds.charset_size + 1, num_emb_out=100, num_classes=21).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = SGD(m.parameters(), lr=0.01, momentum=0.9, nesterov=True)

epochs = 5

print_interval = 1000

for epoch in range(epochs):
    train_losses = []
    for i, (mb, labels, ns) in enumerate(dl):
        m.train()
        optimizer.zero_grad()
        mb, labels = mb.to(device), labels.to(device)
        out = m(mb)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if i % print_interval == 0:
            print("Ave Loss on Epoch {} Batch {}: {}".format(epoch + 1, i + 1, sum(train_losses)/len(train_losses)))

print(labels)
print(torch.nn.functional.softmax(out, 1).max(dim=1)[1])
