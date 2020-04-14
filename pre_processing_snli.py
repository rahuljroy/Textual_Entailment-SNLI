
# Importing Libraries

import numpy as np
import pandas as pd

from google.colab import drive
drive.mount('/content/gdrive')

# Load the datasets - train, test and validation

# data = pd.read_csv('/Dataset/snli_1.0_train.txt', sep='\t')
#data = pd.read_csv('/Dataset/snli_1.0_test.txt', sep='\t')
data = pd.read_csv('/Dataset/snli_1.0_dev.txt', sep='\t')

data.dropna(inplace=True, subset=['sentence2']) # Removes the rows where sentence2 is empty

blanks =[]

for i,x,y in data[['sentence1','sentence2']].itertuples():
    if (x.isspace() or y.isspace()):
        blanks.append(i)

if (len(blanks) != 0):
    data.drop(blanks, inplace=True)

# data_new = data[['sentence1','sentence2']]
sentence1 = data[['sentence1']]
sentence2 = data[['sentence2']]

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    # words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    # words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    return words

new_sentence1=[]
for i, sentence in sentence1.itertuples():
  # if i<100:
    words = nltk.word_tokenize(sentence)
    words = normalize(words)
    words = TreebankWordDetokenizer().detokenize(words)
    # print(words)
    new_sentence1.append(words)

new_sentence2 = []
for i, sentence in sentence2.itertuples():
  # if i<100:
    words = nltk.word_tokenize(sentence)
    words = normalize(words)
    words = TreebankWordDetokenizer().detokenize(words)
    # print(words)
    # sentence2.rename(index={sentence:words},inplace=True)
    new_sentence2.append(words)

from pandas import DataFrame
new_sentence1 = DataFrame({'sentence1':new_sentence1})
new_sentence2 = DataFrame({'sentence1':new_sentence2})

data['sentence1'] = new_sentence1
data['sentence2'] = new_sentence2

# Save the file as a csv

# data.to_csv('/Dataset/snli_1.0_train_Processed.csv')
#data.to_csv('/Dataset/snli_1.0_test_Processed.csv')
data.to_csv('/Dataset/snli_1.0_dev_Processed.csv')