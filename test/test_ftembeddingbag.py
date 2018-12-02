from torch.nn import EmbeddingBag
#from torch.nn.modules.sparse import EmbeddingBag
import numpy as np
import torch
import torch.nn as nn
import random
import string
import time
from data.europarl import EUROPARLFT
from lid.ftModel import FTModel

ft_model_path = "/home/david/Programming/data/WMT/europarl/models/europarl_lid.model.min5.bin"
basedir = "/home/david/Programming/data/WMT/europarl"

#embbag = load_model(ft_model_path)
#input_matrix = embbag.get_input_matrix()
#num_emb, emb_dim = input_matrix.shape
#embbag = nn.EmbeddingBag(num_emb, emb_dim)
#embbag.weight.data.copy_(torch.from_numpy(input_matrix))

sentence = "Der König trägt keine Kleidung ."

ds = EUROPARLFT(basedir, split="dev_train", fasttext_model_path = "models/europarl_lid.model.min5.bin",
                use_spaces = False)
dl = torch.utils.data.DataLoader(ds, batch_size=2)

m = FTModel()

for mb, labels in dl:
    print(mb.size(), labels.size())
    out = m(mb)
    print(out.size())
    break

#word_ids = [word_labeler[w] for w in sentence if w in word_labeler]
#word_ids = torch.from_numpy(np.concatenate(word_ids)).reshape(1, -1)
#out = embbag(word_ids)
#word_id_finder = m.get_words(include_freq = False)
#print([item for sublist in "Der König trägt keine Kleidung .".split(" ") for item in (sublist, "</s>")][:-1])
