# Fellowship.ai Write Up Notes

## Motivation  

I chose the NLP task as my challenge.  I have more experience with audio and video, but wanted to learn more about NLP.  I have read a quite a bit about NLP because sequence to sequence models that I adapted to audio were often designed to work with NLP.  So I have read papers on the subject, but hadn't done much applied work myself.  

## Assignment  

The assignment is to use the European Parliament text dataset (europarl) to train a classifier to do language identification (LID) on a test set of 21 different European languages.  The challenges to solve this problem are the size of the dataset, the multilingual nature of the problem, and the variable length of the input.  Thus I needed to create a model would take a line of text as input and output one of 21 languages.  Summarized below:

Input: line text of unknown length and language
Output: 1 of 21 languages

## Dataset  
The europarl is distributed as a single gzipped tarfile that contains text files with XML annotated files in a directory structure split by language.  In total, there are 187,072 files.  Each line contained either XML or target text / sentence (not all of the lines are full sentences, but for simplicities sake, I may use "sentence" and "line" interchangeably).  Importantly, not all the lines end in a traditional stop character or punctuation.  A sample of one of these files is shown below:

```sh
head txt/de/ep-09-04-21-023.txt
<CHAPTER ID="023">
Erhaltung der Fischereiressourcen durch technische Maßnahmen (Aussprache)
<SPEAKER ID="212" NAME="Präsident">
Als nächster Punkt folgt der Bericht von Herrn Visser im Namen des Ausschusses für Fischerei über den Vorschlag für eine Verordnung des Rates über die Erhaltung der Fischereiressourcen durch technische Maßnahmen - C6-0282/2008 -.
<SPEAKER ID="213" NAME="Carmen Fraga Estévez" LANGUAGE="ES">
Herr Präsident! Ich möchte dem Berichterstatter danken für die Hervorhebung einiger der Hauptbedenken, die sich aus dem Vorschlag der Kommission ergeben.
```

The test dataset is a single text file with each line beginning with the label and a tab character, which is followed by the sentence.  A sample is shown below:

```sh
head test/europarl.test
bg	"Европа 2020" не трябва да стартира нов конкурентен маратон и изход с приватизация.
bg	(CS) Най-голямата несправедливост на сегашната обща селскостопанска политика е фактът, че субсидиите се разпределят неравностойно и несправедливо между старите и новите държави-членки.
bg	(DE) Г-жо председател, г-н член на Комисията, по принцип съм против въвеждането на нови данъци.
bg	(DE) Г-н председател, бих искал да започна с коментар на казаното от члена на Комисията Димас.
bg	(DE) Г-н председател, въпросът за правата на човека и зачитането на правата на малцинствата е постоянен източник на противоречие в продължение на години, ако не и на десетилетия, в отношенията между Европейския съюз и Китайската народна република.
```

The two different formats will require slightly different preprocessing to get them into one format to use with my classifier.  

I also ran some exploratory statistics on both the train set and the test set.  Such as number of total unique words, character frequency and count, total line count, and min/max characters per line.   

## Data Preprocessing

I decided to try the [fastText](https://github.com/facebookresearch/fasttext) library, which as the README states "is a library for efficient learning of word representations and sentence classification."  The input for this library is a single text file with each line beginning with the label preceded by the text `__label__` and then the training text on the remainder of the line. Notably, the original europarl text is not valid XML and all XML annotations occupy a single line with no other information on these XML lines.  To transform this dataset into a usable dataset, I preprocessed the dataset as follows:

1) Recursively read all text files.
2) Remove lines with XML
3) Tokenize each line with a simple tokenizer
4) Write each line with the corresponding label to a single text file

First I loaded each file and grabbed the label from the directory structure.  For each line, a simple check to see if it began with a "<" character was used to check for the presence of XML.  If the line was not XML, then it was tokenized (I'll return to this in a moment) and written to a single large file called `europarl.tokenized.all`.  Each line began with the label in the form `__label__[lang]` (i.e. `__label__de`), a space, and then the sentence.  

At this point, I had to make a decision regarding how to tokenize each line.  Many tokenizers are language specific, but in the test set, I wouldn't know the language at the time of preprocessing.  Thus I decided to use a simple tokenizer whose main purpose was to split punctuation from words.  For example, the string "ergeben." becomes "ergeben ."; however, some common punctuation, such as contractions ("can't" or "gibt's") remain untouched.  Since this corpus is multilingual, I wanted to keep the tokenization as simple as possible.  I ended up choosing the `TweetTokenizer` from the popular python package `nltk`.  Also of note, I attempted to train a classifier on the untokenized data and the results were still very good.  Tokenization did improve the performance, so the final model used tokenized inputs.  

Another choice that I made here was to create a single file instead of recreating the directory and file structure of the original corpus.  The file comes in at 4.8GB and loading this entire file into memory was relatively computationally expensive.  However, loading a single file was a lot less I/O intensive than loading many small files.  In general, I traded off higher disk space usage for lower I/O and memory usage.  To this end, I shuffled this file on disk rather than at training time.  I also split it into a training and validation set on disk.  Due to the massive size of the dataset, I decided to forgo such techniques as n-fold cross validation, extensive hyperparameter searches, nor did I worry too much about overfitting.  With smaller datasets, these concerns are more relevant or computationally inexpensive, but the size of the dataset itself would mitigate most of these concerns.  Especially regarding overfitting, I didn't completely disregard it, but at this stage, I kept it in mind without having it play a role in every decision that I made.  In the end, I never saw any overfitting problems so I didn't do more complex dataset augmentation, such as class balancing.  Thought about it, but didn't do it.  

I will say most of my work was done in this stage.  While model building often is seen as the exciting part of machine learning, I find that data preprocessing and data analysis are far more important.  Understanding the data heavily influences which models would be appropriate for the final task.

## Model and Analysis  

I have primarily worked with audio or video data so I intentionally chose an NLP task to work with something new.  The model classes that I considered were character embedding model, word embedding models, and RNN models.  I looked for pretrained word embeddings, but couldn't find any multilingual embeddings.  Most embeddings required one to know the language of the word beforehand.  I work with PyTorch quite a bit and known that Facebook Research has done quite a bit with unsupervised multilingual language embeddings.  

As stated before, the first model that I wanted to try was fastText word embeddings.  FastText uses a mean embedding bag followed by a single classification layer with sigmoid activations and a softmax negative log likelihood loss function.  This would not even be considered deep learning because only one hidden layer is used.  But the results are difficult to beat.  

I build the library and tried the default settings on all my training data.  The results were pretty amazing.  Training took about 8 minutes on my laptop and resulted in a 99.7% classification accuracy on the test set.  The one problem was that the model was 1.9GB.  However fastText includes a quantizer, which trims the model's vocabulary, quantizes the model, and optionally retrains the quantized model.  Again, using the default settings in the example included in the repo, the model size went from 1.9GB to 6.9MB and the accuracy remained at 99.7%!  That's a model that can run on almost any device with 99.7% accuracy.  I hadn't even used the training and validation splits.  This is bad machine learning, but it worked.

Honestly, if I were working for a company, I would take my quantized model and call it a day.  But I never know if these challenges are designed to show creativity, proper procedures, how one deals with problems when they arise, etc.  Obviously, using an out of the box model that works so well wouldn't give you a great idea of what I could do.  

So I decided to take these word embeddings are create my own classification layer in PyTorch.  FastText has python bindings, but to use the fastText embeddings in my own network, I had to not only reconstruct the model, but also the data loader and data processing pipeline.  First, I needed to extract the word dictionary from fastText and build a data loader that encoded text into lists of word indexes.  Then I extracted the word embedding matrix into a PyTorch embedding bag and classification layer to build a network that accepted a list of word indexes and outputted a language class.  The embedding bag layer takes a list of indexes, one-hot encodes them, adds these one-hot encoded vectors, multiplies the summed vector by the embedding matrix and divides the result by the sum of the summed vector (i.e. the number of words in the sample).  This gives us a vector of constant size as the representation of each line of text, regardless of the length of the text.  

One advantage of this model is the simplicity of the input into the classification model.  Specifically, it transforms a variable length text input into a constant length vector.  The disadvantage is that the embedding bag does not take into account the order of the words.  For a language understanding or text generation model, this could prove problematic, but since languages generally have unique words as long as our word vocabulary is sufficiently large, language identification should be attainable without taking into account word order.  Additionally, the constant length vector requires far less computational power than more complex models such as RNN-based models.  Simpler is better.  

To cutdown the size of the non-quantized model, I trained a fastText model with a smaller vocabulary by only including words that occurred a minimum of 50 times.  This reduced the vocabulary size from 4.6 million to about 400 thousand and reduced the model to 166MB.  I used this reduced model as the basis for my PyTorch model.  I then tried models with two or three hidden layers of sizes `[1000, 200]` and `[1000, 500, 200]`, sigmoid and leaky relu activations, and batch norm, layer norm, and no normalization.  Ultimately, the model with layer sizes of `[1000, 200]`, leaky relu, and layer normalization produced the best results.  The results were positive, but I did not improve on the results of the default fastText model (99.7% vs. 99.5%).  Thus due to simplicity and size, I would go with the quantized default fastText model, which achieved an amazing 99.7% accuracy on the test set.

## Thoughts  

I am very pleased with these results.  I feel like the work done in preprocessing the dataset allowed such a simple model to perform so well.  I believe that understanding the data is an under-rated part of machine learning.  Bigger and bigger models seem to make headlines, but simple models with well-formed inputs can perform excellently as well.  If this were something like a Kaggle competition, one could explore model ensembling to improve the results, but to the detriment of data pipeline and model complexity.  My current solution runs on a CPU and one could retrain the model with minimal effort.  

Additionally, the training dataset and testing dataset seem to have different characteristics.  Notably, the length of the training samples and the length of the test samples are vastly different and the character composition of the two sets also varies greatly.  One could do a deeper analysis of of these differences and possibly tune the training set to get better results.  Depending on the goals of the project, this may be a worthwhile endeavor, but certainly time consuming.  
