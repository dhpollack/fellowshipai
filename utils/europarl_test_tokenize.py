import os
from nltk.tokenize import TweetTokenizer

basedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
testdir = os.path.join(basedir, "test")
testfn = "europarl.test"
labelsfn = "europarl.tokenized.test"

tokenizer = TweetTokenizer()

with open(os.path.join(testdir, testfn), "r") as f:
  data = f.readlines()

def gen_labels_sentences(data):
  for sentence in data:
    label_pos = sentence.find("\t")
    label_str = "__label__" + sentence[:label_pos]
    sentence = label_str + " " + sentence[label_pos + 1:]
    sentence = " ".join(tokenizer.tokenize(sentence)) + "\n"
    yield sentence

with open(os.path.join(testdir, labelsfn), "w") as f:
  f.writelines(gen_labels_sentences(data))
