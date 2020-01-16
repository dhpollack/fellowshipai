import os
import pickle

basedir = "/home/david/Programming/data/WMT/europarl"
europarl_path = os.path.join(basedir, "txt_noxml/europarl.tokenized.all")
charset_pickle_path = os.path.join(basedir, "txt_noxml/charset.pkl")

charset = set([])
labels = set([])

with open(europarl_path, "r") as f:
  for l in f.readlines():
    label, sentence = l.strip().split(" ", 1)
    charset.update(sentence)
    labels.update([label])

charset_dict = {c: i for i, c in enumerate(sorted(list(charset)))}
labels_dict = {l: i for i, l in enumerate(sorted(list(labels)))}

with open(charset_pickle_path, "wb") as f:
  pickle.dump([charset_dict, labels_dict], f)

print(charset_dict, labels_dict)
