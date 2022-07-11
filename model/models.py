import torch
import pandas as pd
import math
import random
import numpy as np
import sklearn as sk
from sklearn.model_selection import KFold
from transformers import BertTokenizer
from torch import nn
from transformers import BertForSequenceClassification, AdamW, BertConfig
import pdb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import device, cuda
import os
import csv

#from .models import CNNClassifier, save_model, load_model
#from .utils import accuracy, load_data
import torch.utils.tensorboard as tb
from torch import device, cuda, manual_seed
from tqdm import tqdm
from torch.optim import SGD, Adam
import numpy as np
import time
import pdb

class BERT_FFNN(torch.nn.Module):
    def __init__(self, base_model, num_classes=3, base_model_output_size=768, dropout=0.1, freeze=False, hidden_layer_size=768):
        super().__init__()
        # TODO: change bert layer to GloVE
        self.base_model = base_model
        #pdb.set_trace()
        if freeze:
            print("Freezing BERT layers")
            # Only freeze BERT layers (not classifier)
            for param in self.base_model.bert.parameters(): 
                param.requires_grad = False

        #self.base_model.eval()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_size, num_classes)
        )
        

    def forward(self, input_):
        #pdb.set_trace()
        x,  attention_mask = input_
        hidden_states = self.base_model(x, attention_mask=attention_mask)[1]
        
        token_vecs = hidden_states[-2][0]
        
        sentence_embedding = torch.mean(token_vecs, dim=0)
        
        #pdb.set_trace()
        #final_features = torch.cat((sentence_embedding, num_feature), 0)

        return self.classifier(sentence_embedding)

class RNNWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, word_embeddings, device, sequence_length=1, num_layers=1, bidirectional=True, dict_size=32):
        super(RNNWrapper, self).__init__()
        #self.embeddings = nn.Embedding(dict_size, input_size)
        self.device = device
        self.indexer = word_embeddings.word_indexer
        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(word_embeddings.vectors), padding_idx=0, freeze=True)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        
        if bidirectional:
            self.classification = nn.Linear(2*hidden_size, num_classes)
        else:
            self.classification = nn.Linear(hidden_size, num_classes)
        #self.softmax = nn.LogSoftmax(dim=2)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        #nn.init.xavier_uniform_(self.rnn.bias_hh_l0)
        #nn.init.xavier_uniform_(self.rnn.bias_ih_l0)

    def forward(self, x):
        
        embedding = self.embeddings(x)

        # change to torch zeros
        if self.bidirectional:
            init = (torch.from_numpy(np.zeros((self.num_layers*2, x.shape[0], self.hidden_size))).float().to(self.device),
                    torch.from_numpy(np.zeros((self.num_layers*2, x.shape[0], self.hidden_size))).float().to(self.device))
        else:
             init = (torch.from_numpy(np.zeros((self.num_layers, x.shape[0], self.hidden_size))).float().to(self.device),
                    torch.from_numpy(np.zeros((self.num_layers, x.shape[0], self.hidden_size))).float().to(self.device))   
        

        output, (last_hidden_state, last_cell_state) = self.rnn(embedding, init)

        output = output[:, -1, :]

        return self.classification(output).squeeze(0)

    def form_input(self, x) -> torch.Tensor:
        out = torch.zeros(self.padding).long()

        for i in range(len(x)):
            word_idx = self.indexer.index_of(x[i])
            out[i] = word_idx if word_idx != -1 else 1
        return out

def save_model(model, name):
    from torch import save
    from os import path
    if isinstance(model, BERT_FFNN):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'bert_ffnn_{}.th'.format(name)))
    if isinstance(model, RNNWrapper):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'rnn_{}.th'.format(name)))
    raise ValueError("model type '%s' not supported!"%str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = BERT_FFNN()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'bert_ffnn.th'), map_location='cpu'))
    return r