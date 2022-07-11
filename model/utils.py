import torch
import pandas as pd
import math
import random
import numpy as np
import sklearn as sk
from copy import deepcopy
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
import itertools
import matplotlib.pyplot as plt

#from .models import CNNClassifier, save_model, load_model
#from .utils import accuracy, load_data
import torch.utils.tensorboard as tb
from .pdtb_relations import PDTB_COLUMNS, COLUMN_LABELS
from torch import device, cuda, manual_seed
from tqdm import tqdm
from torch.optim import SGD, Adam
import numpy as np
import time
from nltk import word_tokenize
import regex

class RNNGloveDataset(Dataset):
    def __init__(self, dataset_path, indexer=None, sequence_length=896):
        self.inputMatrix = []
        df = pd.read_csv(dataset_path)
        labels = []
        articles = []
        dimension_dict = {"From the Center": 0, "From the Right": 1, "From the Left": 2}

        for _,row in df.iterrows():
            articles.append(row["content"])
            labels.append(dimension_dict[row["allsides_bias"]])

        self.input_ids = np.zeros((len(articles), sequence_length), dtype = int)

        if indexer == None:
            print("No indexer passed")
            exit()

        for i, article in enumerate(articles):
            #   (1) Tokenize the sentence.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`

            article_int = [indexer.index_of(x) for x in word_tokenize(regex.sub("\\n", " ", regex.sub("\.", " ", article.lower())))]
            article_int  = [x if x != -1 else 1 for x in article_int]

            article_len = len(article_int)

            if article_len <= sequence_length:
                zeroes = list(np.zeros(sequence_length-article_len))
                new = zeroes+article_int
            elif article_len > sequence_length:
                new = article_int[0:sequence_length]
            
            self.input_ids[i,:] = np.array(new)

        #self.input_ids = torch.cat(input_ids, dim=0)
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]       


class PowerAnalysisGloveDataSet(Dataset):
    def __init__(self, dataset_path, indexer=None, dimension="fiscal", sequence_length=896):
        self.inputMatrix = []
        df = pd.read_csv(dataset_path)
        labels = []
        paragraphs = []
        dimension_dict = {"mixed_val": 0, "moderate_val_right": 1, "moderate_val_left": 2, "extreme_val_right": 1, "extreme_val_left": 2}
        self.num_paragraphs = 6

        for _,row in df.iterrows():
          for i in range(self.num_paragraphs):
            if row["Answer.{}_topic_{}".format(dimension, i)] == "default":
              continue
            elif row["Original.p{}".format(i)] != "empty":
              paragraphs.append(row["Original.p{}".format(i)])
              labels.append(dimension_dict[row["Answer.{}_topic_{}".format(dimension, i)]])
              #num_features.append(process_data(row, dimension, i))

        articles = paragraphs

        self.input_ids = np.zeros((len(articles), sequence_length), dtype = int)

        if indexer == None:
            print("No indexer passed")
            exit()

        for i, article in enumerate(articles):
            #   (1) Tokenize the sentence.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`

            article_int = [indexer.index_of(x) for x in word_tokenize(regex.sub("\\n", " ", regex.sub("\.", " ", article.lower())))]
            article_int  = [x if x != -1 else 1 for x in article_int]

            article_len = len(article_int)

            if article_len <= sequence_length:
                zeroes = list(np.zeros(sequence_length-article_len))
                new = zeroes+article_int
            elif article_len > sequence_length:
                new = article_int[0:sequence_length]
            
            self.input_ids[i,:] = np.array(new)

        #self.input_ids = torch.cat(input_ids, dim=0)
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]  


class PowerAnalysisDataSet(Dataset):
    def __init__(self, dataset_path, dimension="fiscal", tokenizer=None):
        self.inputMatrix = []
        df = pd.read_csv(dataset_path)
        self.num_paragraphs = 6
        paragraphs = []
        labels = []
        dimension_dict = {"mixed_val": 0, "moderate_val_right": 1, "moderate_val_left": 2, "extreme_val_right": 1, "extreme_val_left": 2}
        author_dict = {"mixed_val": 0, "moderate_val_right": 1, "moderate_val_left": 2, "default":-1}
        for _,row in df.iterrows():
          for i in range(self.num_paragraphs):
            if row["Answer.{}_topic_{}".format(dimension, i)] == "default":
              continue
            elif row["Original.p{}".format(i)] != "empty":
              paragraphs.append(row["Original.p{}".format(i)])
              labels.append(dimension_dict[row["Answer.{}_topic_{}".format(dimension, i)]])
              #num_features.append(process_data(row, dimension, i))
         
        articles = paragraphs

        input_ids = []
        attention_masks = []

        if tokenizer == None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # For every sentence...
        for i, article in enumerate(articles):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                                article,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 512,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                truncation = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                          )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        self.input_ids = torch.cat(input_ids, dim=0)
        self.attention_masks = torch.cat(attention_masks, dim=0)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

def load_data(dataset_path,tokenizer=None, dimension="fiscal", num_workers=4, batch_size=16):
    dataset = PowerAnalysisDataSet(dataset_path, dimension, tokenizer=tokenizer)
    return dataset, DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)

def load_data_pretraining_glove(dataset_path,indexer=None, num_workers=4, batch_size=16, sequence_length=896):
    dataset = RNNGloveDataset(dataset_path, indexer=indexer, sequence_length=sequence_length)
    return dataset, DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)

def load_data_glove(dataset_path,indexer=None, num_workers=4, dimension="fiscal", batch_size=16, sequence_length=896):
    dataset = PowerAnalysisGloveDataSet(dataset_path, indexer=indexer, dimension=dimension,  sequence_length=sequence_length)
    return dataset, DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

def plot_confusion_matrix(cm, class_names, normalize=False):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def process_data(row_data, dimension, p):
    discourse_annotations = str(row_data["Discourse.p" + str(p)])
    if discourse_annotations == "nan" or discourse_annotations == "NaN":
        return [0,0,0,0]

    discourse_annotations = discourse_annotations.split('\n')
    for k in range(len(discourse_annotations)):
        if discourse_annotations[k].count("|") == 48:
            discourse_annotations[k] = discourse_annotations[k][:-1]

    df_cols = deepcopy(PDTB_COLUMNS)
    df = pd.DataFrame([x.split('|') for x in discourse_annotations], columns=df_cols)

    relation_count = {"Expansion": 0., "Contingency": 0., "Comparison": 0., "Temporal": 0.}
    for j in df.index:
        relation_type = df[COLUMN_LABELS["relation type"]][j]
        full_sense = df[COLUMN_LABELS["level-1-label"]][j]

        if pd.isnull(full_sense):
            continue

        level_1_sense = full_sense.split(".")[0]
        relation_count[level_1_sense] += 1

    return [relation_count["Expansion"], relation_count["Contingency"], relation_count["Comparison"], relation_count["Temporal"]]


class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]