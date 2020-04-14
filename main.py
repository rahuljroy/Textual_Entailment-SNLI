# Import Libraries
import numpy
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from torch.autograd import Variable

import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import sys
from torchtext import data
from torchtext import datasets

# Parameters of LSTM
batch_size = 512
embedding_dim = 300
dropout_ratio = 0.2
hidden_dim = 256
epochs = 10
lr = 0.001
bidirect = True
combine = 'cat'
opt_name = 'Adam'

# Setting device to CPU
device = torch.device('cpu')

# Parameters of TFIDF_Logistic_Reg
max_iter = 300
l1_ratio = None
tol = 0.001

# Importing the classes for network and the dataset

from bilstm import BiLSTM, SNLI
# from bilstm import SNLI


def test(model, dataset):
    model.eval()
    dataset.test_iter.init_epoch()

    y_gt_label = []
    y_pred_label = []

    correct = 0
    total = 0
    n_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset.test_iter):

            prediction = model(batch)
            yt_pred = Variable(prediction)
            yt_pred = F.softmax(yt_pred, dim=1)
            yt_pred_label = torch.argmax(yt_pred, dim=1)

            yt_gt = Variable(batch.label)

            loss = F.cross_entropy(prediction, batch.label)

            correct += (torch.max(prediction, 1)[1].view(batch.label.size()) == batch.label).sum().item()
            total += batch.batch_size
            n_loss += loss.item()

            # Add the labels
            y_gt_label += list(yt_gt.numpy())
            y_pred_label += list(yt_pred_label.numpy())
    
    test_loss = n_loss/total;
    test_acc = (correct/total) * 100.

    return test_loss, test_acc, y_gt_label, y_pred_label

if __name__ == "__main__":

    # Getting the test data from pytorch

    dataset = SNLI(batch_size, device)
    out_dim = dataset.out_dim()
    vocab_size = dataset.vocabulary_size()

    # from google.colab import drive
    # drive.mount('/content/gdrive')
    
    # Testing the LSTM model

    # Loading the model using the parameters needed
    filename = "Models/LSTM/" + '{0}_{1}_{2}_{3}_{4}_{5}_{6}_bidirect.pt'.format(batch_size, embedding_dim, dropout_ratio, hidden_dim, epochs, opt_name, lr)
    model = BiLSTM(vocab_size, embedding_dim, dropout_ratio, hidden_dim, out_dim, bidirect)
    model.to(device)
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    test_loss, test_accuracy, gt, pred = test(model, dataset)
    # print("Test loss = {}, Test accuracy = {}".format(test_loss, test_accuracy))

    # Writing the output from LSTM onto a text file

    labels = ['entailment', 'contradiction', 'neutral']
    with open("LSTM.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(test_loss))
        f.write("Accuracy on Test Data : {}\n".format(test_accuracy))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(labels[gt[idx]], labels[pred[idx]]))

    # Testing the TF-IDF Logistic Regression

    # Loading the datatset - preprocessed or not
    # data_test = pd.read_csv('/content/gdrive/My Drive/Subjects/Deep_Learning_CSA250/DL_Project3/Dataset/snli_1.0_test_Processed.csv')
    data_test = pd.read_csv('Dataset/snli_1.0_test.txt', sep = '\t')

    test_sent1 = data_test['sentence1']
    test_sent2 = data_test['sentence2']
    y_pred = data_test['gold_label']

    # Loading the Logistic regression model and the vectorizers

    filename = "Models/TFIDF/" + '{0}_{1}_{2}_new.pkl'.format(max_iter, l1_ratio, tol)
    filename1 = 'Models/TFIDF/count_vect.pkl'
    filename2 = 'Models/TFIDF/tfidf_transformer.pkl'

    # count vectorizer and the tf-idf transformer
    count_vect = joblib.load(filename1)
    tfidf_transformer = joblib.load(filename2)

    # Transform the test data using the vectorizers

    test_sent1_vect = count_vect.transform(test_sent1)
    test_sent2_vect = count_vect.transform(test_sent2)

    test_sent1_tfidf = tfidf_transformer.transform(test_sent1_vect)
    test_sent2_tfidf = tfidf_transformer.transform(test_sent2_vect)

    # X_test = sparse.csc_matrix(test_sent1_tfidf).multiply(sparse.csc_matrix(test_sent2_tfidf))
    X_test = test_sent1_tfidf-test_sent2_tfidf

    # Logistic Regression Model

    lr_reg_saved = joblib.load(filename)  
    predictions = lr_reg_saved.predict(X_test)

    from sklearn import metrics

    # Writing the output from the Logistic Regression model onto a text file

    with open("TFIDF_Log_Regression.txt", 'w') as f:
        # f.write("Loss on Test Data : {}\n".format(test_loss))
        f.write("Accuracy on Test Data : {}\n".format(metrics.accuracy_score(y_pred, predictions)))
        f.write("gt_label,pred_label \n")
        for idx in range(len(predictions)):
            f.write("{},{}\n".format(y_pred[idx], predictions[idx]))

    # from sklearn import metrics
    # from sklearn.metrics import confusion_matrix, classification_report

    # print(metrics.accuracy_score(y_pred, predictions))
    # print(confusion_matrix(y_pred, predictions))
    # print(classification_report(y_pred, predictions))