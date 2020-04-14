
import numpy as np
import pandas as pd

# Parameters for the logistic regression model

max_iter = 300
l1_ratio = None
tol = 0.001

# from google.colab import drive
# drive.mount('/content/gdrive')

# Loading the training data - processed or not

# data = pd.read_csv('Dataset/nli_1.0_train_Processed.csv')
data = pd.read_csv('Dataset/snli_1.0_train.txt', sep = '\t')

data.dropna(inplace=True, subset=['sentence2']) # Removes the rows where sentence2 is empty

blanks =[]

for i,x,y in data[['sentence1','sentence2']].itertuples():
    if (x.isspace() or y.isspace()):
        blanks.append(i)

if (len(blanks) != 0):
    data.drop(blanks, inplace=True)

complete_sentences = pd.concat((data['sentence1'], data['sentence2']), axis=0)

vec_len = len(complete_sentences)/2

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count_vect = CountVectorizer(input='content',
    encoding='utf-8',
    decode_error='strict',
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern='(?u)\\b\\w\\w+\\b',
    ngram_range=(1, 1),
    analyzer='word',
    max_df=1.0,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False)

count_vect.fit(complete_sentences)

sent1 = data['sentence1']
sent2 = data['sentence2']
sent1_vect = count_vect.transform(sent1)
sent2_vect = count_vect.transform(sent2)
sent1_vect.shape, sent2_vect.shape

tfidf_transformer = TfidfTransformer()

sent1_tfidf = tfidf_transformer.fit_transform(sent1_vect)
sent2_tfidf = tfidf_transformer.fit_transform(sent2_vect)

tfidf_length, tfidf_width = (sent1_tfidf.shape[0]), (sent1_tfidf.shape[1])

import scipy
from scipy import sparse

# X_train = sparse.csc_matrix(sent1_tfidf).add(sparse.csc_matrix(sent2_tfidf))
X_train = sent1_tfidf-sent2_tfidf

from sklearn.linear_model import LogisticRegression
lr_reg = LogisticRegression(penalty='l2',
    dual=False,
    tol=tol,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='saga',
    max_iter=max_iter,
    multi_class='auto',
    verbose=1,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None)

y_train = data['gold_label']

from sklearn import preprocessing
X_train = preprocessing.scale(X_train, with_mean=False)

lr_reg.fit(X_train, y_train)

import pickle
from sklearn.externals import joblib

filename = "Models/TFIDF/" + '{0}_{1}_{2}_new.pkl'.format(max_iter, l1_ratio, tol)
filename1 = 'Models/TFIDF/count_vect.pkl'
filename2 = 'Models/TFIDF/tfidf_transformer.pkl'

joblib.dump(count_vect, filename1)
joblib.dump(tfidf_transformer, filename2)

joblib.dump(lr_reg, filename)