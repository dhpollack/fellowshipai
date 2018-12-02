import torch
import torch.nn as nn
import torch.utils.data as data
from fastText import load_model
import pickle
import os
import math

class EUROPARL(data.Dataset):
    splits = {"train": "txt_noxml/europarl.tokenized.train",
              "test": "test/europarl.tokenized.test",
              "dev_train": "txt_noxml/europarl.tokenized.trunc.train",
              "dev_test": "test/europarl.tokenized.trunc.test"}
    def __init__(self, basedir, split = "train", precalc_cs_and_ls = "txt_noxml/charset.pkl",
                 max_samples = 10, sample_len = 4, truncate_to=None):

        self.max_samples = max_samples
        self.sample_len = sample_len

        self._load_charset_and_labels(os.path.join(basedir, precalc_cs_and_ls))

        self.data = []
        self.labels = []
        self.sample_broadcaster = torch.arange(self.sample_len, dtype=torch.long).reshape(1, -1)

        with open(os.path.join(basedir, self.splits[split]), "r") as f:
            for s in f.readlines():
                label, sentence = s.strip().split(" ", 1)
                if truncate_to:
                    sentence = sentence[:truncate_to]
                self.labels.append(label)
                self.data.append(sentence)
    def _load_charset_and_labels(self, path):
        with open(path, "rb") as f:
            self.charset, self.label_encoder = pickle.load(f)
        self.charset_size = len(self.charset)
        self.labelset_size = len(self.label_encoder)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        # encode sentence
        sentence = [self.charset[c] for c in self.data[index]]
        # if sentence not long enough then pad with a dummy character
        if len(sentence) < self.sample_len:
            diff = self.sample_len - len(sentence)
            sentence += [self.charset_size] * diff
        # sentence to torch tensor
        sentence = torch.LongTensor(sentence)
        # calculate number of samples to take from sentence
        num_samples = min(self.max_samples, math.ceil(sentence.size(0) / self.sample_len))
        # create matrix of indices to take from the sentence
        sample_idxes = torch.randint(sentence.size(0) - (self.sample_len - 1), (num_samples, 1), dtype=torch.long)
        sample_idxes = sample_idxes + self.sample_broadcaster
        # grab samples
        samples = torch.take(sentence, sample_idxes)  # num_samples x sample_len
        # make labels for samples
        labels = [self.label_encoder[self.labels[index]]] * num_samples
        labels = torch.LongTensor(labels)
        return samples, labels, num_samples
    def shuffle_data(self):
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data[:], self.labels[:] = zip(*combined)

class EUROPARL2(data.Dataset):
    splits = {"train": "txt_noxml/europarl.tokenized.train",
              "test": "test/europarl.tokenized.test",
              "dev_train": "txt_noxml/europarl.tokenized.trunc.train",
              "dev_test": "test/europarl.tokenized.trunc.test"}
    splits_len = {"train": 10088220,
                  "test": 21000,
                  "dev_train": 100,
                  "dev_test": 5}
    def __init__(self, basedir, split = "train", precalc_cs_and_ls = "txt_noxml/charset.pkl",
                 max_samples = 10, sample_len = 4, truncate_to=None):

        self.basedir = basedir
        self.split = split
        self.max_samples = max_samples
        self.sample_len = sample_len
        self.truncate_to = truncate_to
        self.f = open(os.path.join(self.basedir, self.splits[self.split]), "r")

        self.len = 1
        self.update_len = True

        self._load_charset_and_labels(os.path.join(basedir, precalc_cs_and_ls))

        self.data = []
        self.labels = []
        self.sample_broadcaster = torch.arange(self.sample_len, dtype=torch.long).reshape(1, -1)

    def _get_line(self):
        try:
            s = next(self.f)
            return s
        except StopIteration:
            self.f.seek(0)
            s = next(self.f)
            return s
    def _load_charset_and_labels(self, path):
        with open(path, "rb") as f:
            self.charset, self.label_encoder = pickle.load(f)
        self.charset_size = len(self.charset)
        self.labelset_size = len(self.label_encoder)
    def __len__(self):
        return self.splits_len[self.split]
    def __getitem__(self, index):
        # encode sentence
        s = self._get_line()
        label, sentence = s.strip().split(" ", 1)
        if self.truncate_to:
            sentence = sentence[:self.truncate_to]
        sentence = [self.charset[c] for c in sentence]
        # if sentence not long enough then pad with a dummy character
        if len(sentence) < self.sample_len:
            diff = self.sample_len - len(sentence)
            sentence += [self.charset_size] * diff
        # sentence to torch tensor
        sentence = torch.LongTensor(sentence)
        # calculate number of samples to take from sentence
        num_samples = min(self.max_samples, math.ceil(sentence.size(0) / self.sample_len))
        # create matrix of indices to take from the sentence
        sample_idxes = torch.randint(sentence.size(0) - (self.sample_len - 1), (num_samples, 1), dtype=torch.long)
        sample_idxes = sample_idxes + self.sample_broadcaster
        # grab samples
        samples = torch.take(sentence, sample_idxes)  # num_samples x sample_len
        # make labels for samples
        labels = [self.label_encoder[label]] * num_samples
        labels = torch.LongTensor(labels)
        return samples, labels, num_samples
    def shuffle_data(self):
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data[:], self.labels[:] = zip(*combined)

class EUROPARLFT(data.Dataset):
    splits = {"train": "txt_noxml/europarl.tokenized.all",
              "test": "test/europarl.tokenized.test",
              "split_train": "txt_noxml/europarl.tokenized.split.train",
              "split_valid": "txt_noxml/europarl.tokenized.split.valid",
              "dev_train": "txt_noxml/europarl.tokenized.trunc.train",
              "dev_test": "test/europarl.tokenized.trunc.test"}
    splits_len = {"train": 10088220,
                  "test": 21000,
                  "split_train": 9988221,
                  "split_valid": 100000,
                  "dev_train": 100,
                  "dev_test": 5}
    def __init__(self, basedir, split = "train", fasttext_model_path = "models/europarl_lid.model.min5.bin",
                 use_spaces = False, default_char = "</s>"):

        self.basedir = basedir
        self.split = split
        self.use_spaces = use_spaces
        self.default_char = default_char

        self.f = open(os.path.join(self.basedir, self.splits[self.split]), "r")

        fasttext_model = load_model(os.path.join(basedir, fasttext_model_path))
        input_matrix = fasttext_model.get_input_matrix()  # numpy
        num_emb, emb_dim = input_matrix.shape
        self.embbag = nn.EmbeddingBag(num_emb, emb_dim)
        self.embbag.weight.data.copy_(torch.from_numpy(input_matrix))
        self.embbag.eval()

        self.word_dict = {w: i for i, w in enumerate(fasttext_model.get_words(include_freq=False))}
        self.label_dict = {l: i for i, l in enumerate(fasttext_model.get_labels(include_freq=False))}

        self.data = []
        self.labels = []

    def _get_line(self):
        try:
            s = next(self.f)
            return s
        except StopIteration:
            self.f.seek(0)
            s = next(self.f)
            return s
    def __len__(self):
        return self.splits_len[self.split]
    def __getitem__(self, index):
        # encode sentence
        s = self._get_line()
        label, sentence = s.strip().split(" ", 1)
        if self.use_spaces:
            words_with_space = [w for w_space in sentence.split(" ") for w in (w_space, "</s>")][:-1]
            word_ids = [self.word_dict[w] for w in words_with_space if w in self.word_dict]
        else:
            word_ids = [self.word_dict[w] for w in sentence.split(" ") if w in self.word_dict]
        # sentence to torch tensor
        if not word_ids:
            word_ids = [self.word_dict[self.default_char]]
        word_ids = torch.LongTensor(word_ids).reshape(1, -1)
        # make labels for samples
        label_id = torch.tensor(self.label_dict[label]).long()
        with torch.no_grad():
            word_vec_bag = self.embbag(word_ids).squeeze()
        return word_vec_bag, label_id

def europarl_collate_fn(batch):
    sentences, labels, num_samples = zip(*batch)

    return torch.cat(sentences), torch.cat(labels), num_samples

def europarl_fastai_collate_fn(batch):
    sentences, labels, num_samples = zip(*batch)

    return torch.cat(sentences), torch.cat(labels)
