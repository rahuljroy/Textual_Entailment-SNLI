import numpy
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from torch.autograd import Variable

import dill
import sys
from torchtext import data
from torchtext import datasets



class SNLI():
    def __init__(self, batch_size, device):
        self.inputs = data.Field(lower=True, tokenize = None, batch_first = True)
        self.labels = data.Field(sequential = False, unk_token = None, is_target = True)
        self.train, self.val, self.test = datasets.SNLI.splits(self.inputs, self.labels)
        self.inputs.build_vocab(self.train, self.val)
        self.labels.build_vocab(self.train)
        self.train_iter, self.val_iter, self.test_iter = data.Iterator.splits((self.train, self.val, self.test), batch_size = batch_size, device=device)
    
    def vocabulary_size(self):
        return len(self.inputs.vocab)
    
    def out_dim(self):
        return len(self.labels.vocab)
    
    def labels(self):
        return self.labels.vocab.stoi

class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, dropout_ratio, hidden_dim, out_dim, bidirect):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, 300)
        self.dropout = nn.Dropout(p = dropout_ratio)
        self.lstm = nn.LSTM(300, hidden_dim, 3, bidirectional=bidirect)
        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(2*512, 512),
			self.relu,
			self.dropout,
			nn.Linear(512, 256),
			self.relu,
			self.dropout,
			nn.Linear(256, 256),
			self.relu,
			self.dropout,
			nn.Linear(256, out_dim)
        )
        pass

    def forward(self, batch):
        premise_embedding = self.embedding(batch.premise)
        hypothesis_embedding = self.embedding(batch.hypothesis)

        premise_projection = self.relu(self.projection(premise_embedding))
        hypothesis_projection = self.relu(self.projection(hypothesis_embedding))

        encoded_premise, _ = self.lstm(premise_projection)
        encoded_hypothesis, _ = self.lstm(hypothesis_projection)

        premise = encoded_premise.sum(dim=1)
        hypothesis = encoded_hypothesis.sum(dim=1)

        combined = torch.cat((premise, hypothesis), 1)

        return self.out(combined)

def get_optimizer(model, opt_name, lr, l2_penalty, momentum=None):
        if opt_name == 'SGD':
            return optim.SGD(
                model.parameters(), lr, weight_decay=l2_penalty)
        elif opt_name == 'Momentum':
            return optim.SGD(
                model.parameters(), lr=lr, momentum=momentum,
                weight_decay=l2_penalty)
        elif opt_name == 'Nesterov':
            return optim.SGD(
                model.parameters(), lr=lr, momentum=momentum,
                weight_decay=l2_penalty, nesterov=True)
        elif opt_name == 'Adagrad':
            return optim.Adagrad(
                model.parameters(), lr=lr, weight_decay=l2_penalty)
        elif opt_name == 'RMSProp':
            return optim.RMSprop(
                model.parameters(), lr=lr, weight_decay=l2_penalty)
        elif opt_name == 'Adam':
            return optim.Adam(
                model.parameters(), lr=lr, weight_decay=l2_penalty)

def train(model, dataset):
    model.train()
    dataset.train_iter.init_epoch()

    correct = 0
    total = 0
    n_loss = 0

    for batch_idx, batch in enumerate(dataset.train_iter):

        model.optimizer.zero_grad()
        prediction = model(batch)
        loss = F.cross_entropy(prediction, batch.label)

        correct += (torch.max(prediction, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        total += batch.batch_size
        n_loss += loss.item()

        loss.backward()
        model.optimizer.step()
    
    train_loss = n_loss/total;
    train_acc = (correct/total) * 100.

    return train_loss, train_acc

def validate(model, dataset):
    model.eval()
    dataset.val_iter.init_epoch()

    correct = 0
    total = 0
    n_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset.val_iter):

            prediction = model(batch)
            loss = F.cross_entropy(prediction, batch.label)

            correct += (torch.max(prediction, 1)[1].view(batch.label.size()) == batch.label).sum().item()
            total += batch.batch_size
            n_loss += loss.item()
    
    val_loss = n_loss/total;
    val_acc = (correct/total) * 100.

    return val_loss, val_acc

if __name__ == "__main__":

    batch_size = 512
    embedding_dim = 150
    dropout_ratio = 0.2
    hidden_dim = 256
    epochs = 10
    lr = 0.001
    bidirect = True
    combine = 'cat'

    if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device('cuda:{}'.format(0))
    else:
        device = torch.device('cpu')

    print(device)

    opt_name = 'Adam'
    l2_penalty = 0
    momentum = None

    dataset = SNLI(batch_size, device)
    out_dim = dataset.out_dim()
    vocab_size = dataset.vocabulary_size()

    model = BiLSTM(vocab_size, embedding_dim, dropout_ratio, hidden_dim, out_dim, bidirect)
    model.to(device)
    model.optimizer = get_optimizer(model, opt_name, lr, l2_penalty)
    
    vocab_size = dataset.vocabulary_size()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(epochs):

        start = time.time()
        training_loss, training_accuracy = train(model, dataset)
        validation_loss, validation_accuracy = validate(model, dataset)
        stop = time.time()

        train_loss.append(training_loss)
        train_acc.append(training_accuracy)
        val_loss.append(validation_loss)
        val_acc.append(validation_accuracy)

        print("Time: {}, Epoch: {}, Training loss: {}, Training Accuracy: {}, Validation loss: {}, Validation Accuracy: {}".format(stop-start, epoch+1, training_loss, training_accuracy, validation_loss, validation_accuracy))


    filename = "Models/LSTM/" + '{0}_{1}_{2}_{3}_{4}_{5}_{6}_bidirect.pt'.format(batch_size, embedding_dim, dropout_ratio, hidden_dim, epochs, opt_name, lr)
    torch.save(model.state_dict(), filename)
    print(filename + ' saved successfully\n')

    fig = plt.figure()
    plt.plot(train_loss, label = 'Training Loss')
    plt.plot(val_loss, label = 'Validation Loss')
    plt.title("Loss LSTM")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    filename = "Plots/" + '{0}_{1}_{2}_{3}_{4}_{5}_{6}_bidirect.png'.format(batch_size, embedding_dim, dropout_ratio, hidden_dim, epochs, opt_name, lr)
    plt.savefig(filename)
    plt.show()