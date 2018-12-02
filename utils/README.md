# Utils

These utilities will tokenize the europarl dataset.  The following commands were also performed on the dataset:

```sh
shuf [europarl_path]/txt_noxml/europarl.tokenized.all -o [europarl_path]/txt_noxml/europarl.tokenized.all
head -n 100000 [europarl_path]/txt_noxml/europarl.tokenized.all > [europarl_path]/txt_noxml/europarl.tokenized.split.valid
tail -n +100000 [europarl_path]/txt_noxml/europarl.tokenized.all > [europarl_path]/txt_noxml/europarl.tokenized.split.train
```
