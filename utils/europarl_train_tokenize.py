from glob import glob
import os
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer

basedir = "/home/david/Programming/data/WMT/europarl"
txt_dir = os.path.join(basedir, "txt")
noxml_dir = os.path.join(basedir, "txt_noxml")

text_file = os.path.join(noxml_dir, "europarl.tokenized.all")

tokenizer = TweetTokenizer()

save_files = False

if os.path.exists(text_file):
  os.remove(text_file)

def strip_xml_lines(raw_lines):
  return [x for x in raw_lines if x[0] != "<"]

files = glob(os.path.join(txt_dir, "**/*"))
alllines = []
labels = []

for fp_txt in tqdm(files):
  lines = []
  fn_noxml = os.path.basename(fp_txt)
  dn_noxml = os.path.dirname(fp_txt).replace(txt_dir, noxml_dir)
  label = dn_noxml.split("/")[-1]
  path_noxml = os.path.join(dn_noxml, fn_noxml)
  with open(fp_txt, "r", errors="backslashreplace") as f:
    lines = strip_xml_lines(f.readlines())
  if not os.path.exists(dn_noxml):
    os.makedirs(dn_noxml)
  if save_files:
    with open(path_noxml, "w") as f:
      if lines:
       f.writelines(lines)


  with open(text_file, "a+") as f:
    gen = ("__label__" + l + " " + " ".join(tokenizer.tokenize(s)) + "\n" for l, s in zip([label]*len(lines), lines) if s.strip())
    f.writelines(gen)

print("{} files processed".format(len(files)))
