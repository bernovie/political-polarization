import torch
import pandas as pd
import math
import random
import numpy as np
import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizer
from torch import nn
from transformers import BertForSequenceClassification, AdamW, BertConfig
import pdb
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch import device, cuda
import os
import csv
from .models import *
from .utils import *
from .focal_loss import *
from .glove_data import *

import torch.utils.tensorboard as tb
from torch import device, cuda, manual_seed
from tqdm import tqdm
from torch.optim import SGD, Adam
import numpy as np
import time

def train_rnn(args):
    from os import path
    # manual_seed(12313)
    # np.random.seed(0)
    dev = device('cuda' if cuda.is_available() else 'cpu')

    print("Device: {}".format(dev))
    dimension = args.dimension

    if args.pretrain:
        glove_embeddings = read_word_embeddings("./data/glove.6B.{}d-relativized_pretraining.txt".format(args.glove_size))
        all_data, _ = load_data_pretraining_glove("./data/data_budget_nlpcss20.csv", sequence_length=args.seq_length, indexer=glove_embeddings.word_indexer, num_workers=4, batch_size=args.batch_size)
        weights = torch.tensor([0.13004032258064516, 0.3790322580645161, 0.4909274193548387], dtype=torch.float32)
        weights = 1/weights
        weights = weights/weights.sum()
        class_weights = weights.to(dev)
    else:
        glove_embeddings = read_word_embeddings("./data/glove.6B.{}d-relativized.txt".format(args.glove_size))
        _, train_data = load_data_glove("./data/gold_train.csv", dimension=dimension, sequence_length=args.seq_length, indexer=glove_embeddings.word_indexer, num_workers=4, batch_size=args.batch_size)
        _, dev_data = load_data_glove("./data/gold_dev.csv", dimension=dimension, sequence_length=args.seq_length, indexer=glove_embeddings.word_indexer, num_workers=4, batch_size=args.batch_size)
        _, test_data = load_data_glove("./data/gold_test.csv", dimension=dimension, sequence_length=args.seq_length, indexer=glove_embeddings.word_indexer, num_workers=4, batch_size=args.batch_size)

        
        if args.dimension == "fiscal":
            weights = torch.tensor([0.73763441, 0.11182796, 0.15053763], dtype=torch.float32)
        elif args.dimension == "social":
            weights = torch.tensor([0.20675105, 0.28691983, 0.50632911], dtype=torch.float32)
        elif args.dimension == "foreign":
            weights = torch.tensor([0.08940397, 0.3807947 , 0.52980132], dtype=torch.float32)

        weights = 1/weights
        weights = weights/weights.sum()
        class_weights = weights.to(dev)     

    loss_fn = focal_loss(alpha=class_weights, gamma=args.gamma, reduction="mean", device=dev)

    total_dev_accuracy = 0
    total_dev_f1 = 0
    total_test_accuracy = 0
    total_test_f1 = 0
    log_time = '{}'.format(time.strftime('%H-%M-%S'))

        #pdb.set_trace()

    train_logger, valid_logger, test_logger = None, None, None
    log_name = 'lr={}_batch_size={}_optim={}_epochs={}_gamma={}'.format(args.alpha, args.batch_size, args.optim, args.epochs, args.gamma)
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_{}_{}'.format(log_name, log_time)))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_{}_{}'.format(log_name, log_time)))
        test_logger = tb.SummaryWriter(path.join(args.log_dir, 'test_{}_{}'.format(log_name, log_time)))

    if args.load:
        base_model = BertForSequenceClassification.from_pretrained("./model_save", output_hidden_states=True)
        #tokenizer = BertTokenizer.from_pretrained("./model_save")
        model = BERT_FFNN(base_model=base_model)
    else: 
        model = RNNWrapper(int(args.glove_size), args.hidden_size, 3, glove_embeddings, dev, num_layers=args.layers, bidirectional=args.bidirectional, sequence_length=args.seq_length)

    model.to(dev)
    
    if args.optim == "sgd":
        optimizer = SGD(model.parameters(), lr=args.alpha, momentum=0.9, weight_decay=1e-6)
    if args.optim == "adam":
        optimizer = Adam(model.parameters(), lr=args.alpha, weight_decay=1e-6, eps = 1e-8)
    global_step = 0
    
    for epoch in tqdm(range(args.epochs)):
        model.train()
        #pdb.set_trace()

        for batch in train_data:
            input_ids = batch[0].to(dev)
            y = batch[1].to(dev)
            #pdb.set_trace()

            #import pdb; pdb.set_trace()
            y_pred = model(input_ids)
            

            if len(y_pred.shape) == 1 and len(y.shape) == 1:
              y_pred = y_pred.unsqueeze(0)

              #print(y_pred)
              #print(y)

              #import pdb; pdb.set_trace()

            #print("Ypred shape: {} | y shape: {}".format(y_pred.shape, y.shape))

            loss = loss_fn(y_pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad() 

            train_logger.add_scalar('loss_{}'.format(dimension), loss.item(), global_step=global_step)
            global_step += 1

        model.eval()
        accs = []
        y_true_list = torch.tensor([], dtype=torch.uint8)
        y_pred_list = torch.tensor([], dtype=torch.uint8)
        #pdb.set_trace()
        for batch in dev_data:
            input_ids = batch[0].to(dev)
            y = batch[1].to(dev)

            with torch.no_grad():
                y_pred = model(input_ids)
            #pdb.set_trace()
 
            if len(y_pred.shape) == 1:
              y_pred = y_pred.unsqueeze(0)

            y_pred_list = torch.cat((y_pred_list, y_pred.max(1)[1].cpu()))
            y_true_list = torch.cat((y_true_list, y.cpu()))
            accs.append(accuracy(y_pred, y))

        cm = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1, 2])
        valid_logger.add_scalar('accuracy_{}'.format(dimension), sum(accs)/len(accs), global_step=global_step)
        valid_logger.add_scalar('macro_f1_{}'.format(dimension), f1_score(y_true_list, y_pred_list, average='macro'), global_step=global_step)
        valid_logger.add_scalar('weighted_f1_{}'.format(dimension), f1_score(y_true_list, y_pred_list, average='weighted'), global_step=global_step)
        valid_logger.add_figure("confusion_matrix_{}".format(dimension), plot_confusion_matrix(cm, class_names=["mixed", "conservative", "liberal"]), global_step=global_step)
    
    total_dev_accuracy += sum(accs)/len(accs)
    total_dev_f1 += f1_score(y_true_list, y_pred_list, average='macro')
                
    print("Development accuracy: {:.3f}".format(total_dev_accuracy))
    print("Development macro f1: {:.3f}".format(total_dev_f1 ))
    valid_logger.add_scalar("accuracy", total_dev_accuracy, global_step=0)
    valid_logger.add_scalar("macro f1", total_dev_f1, global_step=0)

    model.eval()
    accs = []
    y_true_list = torch.tensor([], dtype=torch.uint8)
    y_pred_list = torch.tensor([], dtype=torch.uint8)
    #pdb.set_trace()
    for batch in test_data:
        input_ids = batch[0].to(dev)
        y = batch[1].to(dev)

        with torch.no_grad():
            y_pred = model(input_ids)
        #pdb.set_trace()
        if len(y_pred.shape) == 1:
          y_pred = y_pred.unsqueeze(0)
          
        y_pred_list = torch.cat((y_pred_list, y_pred.max(1)[1].cpu()))
        y_true_list = torch.cat((y_true_list, y.cpu()))
        accs.append(accuracy(y_pred, y))

    total_test_accuracy += sum(accs)/len(accs)
    total_test_f1 += f1_score(y_true_list, y_pred_list, average='macro')
                
                #print(cm)

    print("Test accuracy: {:.3f}".format(total_test_accuracy))
    print("Test macro f1: {:.3f}".format(total_test_f1))
    test_logger.add_scalar("accuracy", total_test_accuracy, global_step=0)
    test_logger.add_scalar("macro f1", total_test_f1, global_step=0)


    if not args.no_save:
        print("Saving model")
        save_model(model, "batch_16")

    return {"macro_f1":total_test_f1, "accuracy":total_test_accuracy}


def train(args):
    from os import path
    # manual_seed(12313)
    # np.random.seed(0)
    dev = device('cuda' if cuda.is_available() else 'cpu')

    print("Device: {}".format(dev))

    if args.load:
        tokenizer = BertTokenizer.from_pretrained("./model_save")
    else:
        tokenizer = None

    if args.dimension == "fiscal":
        _, train_data = load_data("./data/gold_train.csv", dimension="fiscal", num_workers=4, batch_size=args.batch_size)
        _, dev_data = load_data("./data/gold_dev.csv", dimension="fiscal", num_workers=4, batch_size=args.batch_size)
        _, test_data = load_data("./data/gold_test.csv", dimension="fiscal", num_workers=4, batch_size=args.batch_size)
        #all_data, _ = load_fiscal_data("./data/all_batches_gold.csv", tokenizer=tokenizer, num_workers=4, batch_size=args.batch_size)
        weights = torch.tensor([0.77216917, 0.09140518, 0.13642565], dtype=torch.float32)
        weights = 1/weights
        weights = weights/weights.sum()
        class_weights = weights.to(dev)
    elif args.dimension == "social":
        _, train_data = load_data("./data/gold_train.csv", dimension="social", num_workers=4, batch_size=args.batch_size)
        _, dev_data = load_data("./data/gold_dev.csv", dimension="social", num_workers=4, batch_size=args.batch_size)
        _, test_data = load_data("./data/gold_test.csv", dimension="social", num_workers=4, batch_size=args.batch_size)
        #all_data, _ = load_social_data("./data/all_batches_gold.csv", tokenizer=tokenizer, num_workers=4, batch_size=args.batch_size)
        weights = torch.tensor([0.26785714, 0.27678571, 0.45535714], dtype=torch.float32)
        weights = weights/weights.sum()
        class_weights = weights.to(dev)
    elif args.dimension == "foreign":       
        _, train_data = load_data("./data/gold_train.csv", dimension="foreign", num_workers=4, batch_size=args.batch_size)
        _, dev_data = load_data("./data/gold_dev.csv", dimension="foreign", num_workers=4, batch_size=args.batch_size)
        _, test_data = load_data("./data/gold_test.csv", dimension="foreign", num_workers=4, batch_size=args.batch_size)
        #all_data, _ = load_foreign_data("./data/all_batches_gold.csv",  tokenizer=tokenizer,num_workers=4, batch_size=args.batch_size)
        weights = torch.tensor([0.09785203, 0.40811456, 0.49403341], dtype=torch.float32)
        weights = 1/weights
        weights = weights/weights.sum()
        class_weights = weights.to(dev)

    loss_fn = focal_loss(alpha=class_weights, gamma=args.gamma, reduction="mean", device=dev)

    total_dev_accuracy = 0
    total_dev_f1 = 0
    total_test_accuracy = 0
    total_test_f1 = 0
    log_time = '{}'.format(time.strftime('%H-%M-%S'))


    train_logger, valid_logger, test_logger = None, None, None
    log_name = 'lr={}_batch_size={}_optim={}_epochs={}_gamma={}'.format(args.alpha, args.batch_size, args.optim, args.epochs, args.gamma)
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_{}_{}'.format(log_name, log_time)))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_{}_{}'.format(log_name, log_time)))
        test_logger = tb.SummaryWriter(path.join(args.log_dir, 'test_{}_{}'.format(log_name, log_time)))

    if args.load:
        base_model = BertForSequenceClassification.from_pretrained("./model_save", output_hidden_states=True)
        #tokenizer = BertTokenizer.from_pretrained("./model_save")
        model = BERT_FFNN(base_model=base_model, dropout=args.dropout, freeze=args.freeze, hidden_layer_size=args.hidden_size_bert)
    else: 
        base_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True,
        )
        model = BERT_FFNN(base_model=base_model, dropout=args.dropout, freeze=args.freeze, hidden_layer_size=args.hidden_size_bert, base_model_output_size=768)

    model.to(dev)
    
    if args.optim == "sgd":
        optimizer = SGD(model.parameters(), lr=args.alpha, momentum=0.9, weight_decay=1e-6)
    if args.optim == "adam":
        optimizer = Adam(model.parameters(), lr=args.alpha, weight_decay=1e-6, eps = 1e-8)
    global_step = 0
    
    for epoch in tqdm(range(args.epochs)):
        model.train()
        #pdb.set_trace()

        model.zero_grad() 

        for batch in train_data:
            input_ids = batch[0].to(dev)
            attention_masks = batch[1].to(dev)
            y = batch[2].to(dev)
            #num_features = batch[3].to(dev)
            #pdb.set_trace()
            paragraphs = [ (input_ids[i], attention_masks[i]) for i in range(len(input_ids))]

            for j, vals in enumerate(paragraphs):
                input_id = vals[0]
                attention_mask = vals[1]
                #num_feature = vals[2]
                y_pred = model((input_id.unsqueeze(0), attention_mask.unsqueeze(0)))

                loss = loss_fn(y_pred[None], y[j][None])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() 

                train_logger.add_scalar('loss_{}'.format(args.dimension), loss.item(), global_step=global_step)
                global_step += 1

        model.eval()
        accs = []
        y_true_list = []
        y_pred_list = []
        #pdb.set_trace()
        for batch in dev_data:
            input_ids = batch[0].to(dev)
            attention_masks = batch[1].to(dev)
            y = batch[2].to(dev)
            #num_features = batch[3].to(dev)
            paragraphs = [ (input_ids[i], attention_masks[i]) for i in range(len(input_ids))]


            for j, vals in enumerate(paragraphs):
                input_id = vals[0]
                attention_mask = vals[1]
                #num_feature = vals[2]
                with torch.no_grad():
                    y_pred = model((input_id.unsqueeze(0), attention_mask.unsqueeze(0)))
                #pdb.set_trace()
                y_pred_list.append(y_pred.max(0)[1].cpu())
                y_true_list.append(y[j].cpu())
                accs.append(accuracy(y_pred[None], y[j][None]))

        cm = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1, 2])
        valid_logger.add_scalar('accuracy_{}'.format(args.dimension), sum(accs)/len(accs), global_step=global_step)
        valid_logger.add_scalar('macro_f1_{}'.format(args.dimension), f1_score(y_true_list, y_pred_list, average='macro'), global_step=global_step)
        valid_logger.add_scalar('weighted_f1_{}'.format(args.dimension), f1_score(y_true_list, y_pred_list, average='weighted'), global_step=global_step)
        valid_logger.add_figure("confusion_matrix_{}".format(args.dimension), plot_confusion_matrix(cm, class_names=["mixed", "conservative", "liberal"]), global_step=global_step)
    
    total_dev_accuracy += sum(accs)/len(accs)
    total_dev_f1 += f1_score(y_true_list, y_pred_list, average='macro')
                
                #print(cm)

    print("Development accuracy: {:.3f}".format(total_dev_accuracy))
    print("Development macro f1: {:.3f}".format(total_dev_f1))
    valid_logger.add_scalar("accuracy", total_dev_accuracy, global_step=0)
    valid_logger.add_scalar("macro f1", total_dev_f1, global_step=0)


    model.eval()
    accs = []
    y_true_list = []
    y_pred_list = []
    #pdb.set_trace()
    for batch in test_data:
        input_ids = batch[0].to(dev)
        attention_masks = batch[1].to(dev)
        y = batch[2].to(dev)
        #num_features = batch[3].to(dev)
        paragraphs = [ (input_ids[i], attention_masks[i]) for i in range(len(input_ids))]

        for j, vals in enumerate(paragraphs):
            input_id = vals[0]
            attention_mask = vals[1]
            #num_feature = vals[2]
            with torch.no_grad():
                y_pred = model((input_id.unsqueeze(0), attention_mask.unsqueeze(0)))
            #pdb.set_trace()
            y_pred_list.append(y_pred.max(0)[1].cpu())
            y_true_list.append(y[j].cpu())
            accs.append(accuracy(y_pred[None], y[j][None]))

    total_test_accuracy += sum(accs)/len(accs)
    total_test_f1 += f1_score(y_true_list, y_pred_list, average='macro')
                
    print("Test accuracy: {:.3f}".format(total_test_accuracy))
    print("Test macro f1: {:.3f}".format(total_test_f1))
    test_logger.add_scalar("accuracy", total_test_accuracy, global_step=0)
    test_logger.add_scalar("macro f1", total_test_f1, global_step=0)

    if not args.no_save:
        print("Saving model")
        save_model(model, "batch_16")

    return {"macro_f1":total_test_f1, "accuracy":total_test_accuracy}

if __name__ == "__main__":
    import argparse
    
    # manual_seed(12313)
    # np.random.seed(0)
    dev = device('cuda' if cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description="Model parameters")

    parser.add_argument('--log_dir', default="./logs")
    parser.add_argument('-a', '--alpha', type=float, default=1e-4, help="learning rate")
    parser.add_argument('-e', "--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('-o', "--optim", choices=['sgd', 'adam'], default="adam", help="optimizer type")
    parser.add_argument('-l', "--load", action="store_true", default=False, help="continue training previously saved model?")
    parser.add_argument('-s', "--no_save", action="store_true", default=False, help="don't save the model")
    parser.add_argument('-nl', "--layers", type=int, default=2, help="Number of layers for LSTM")
    parser.add_argument('-b', "--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument('-d', "--dimension", type=str, default="fiscal", help="dimension to train the model with")
    parser.add_argument('-g', "--gamma", type=float, default=2.0, help="gamma for focal loss")
    parser.add_argument('-m', "--model", choices=['BERT', 'LSTM'], default="BERT", help="model to use")
    parser.add_argument('-sl', "--seq_length", type=int, default=896, help="Sequence Length for LSTM")
    parser.add_argument('-hs', "--hidden_size", type=int, default=256, help="Hidden Size for LSTM")
    parser.add_argument('-p', "--pretrain", action="store_true", default=False, help="pretrain model?")
    parser.add_argument('-c', "--clip_grad", type=int, default=5, help="Number to clip gradients of LSTM to")
    parser.add_argument('-bi', "--bidirectional", action="store_true", default=False, help="use bidirectional LSTM?")
    parser.add_argument('-gs', "--glove_size", choices=['50', '100'], default="50", help="GlOVe embedding size")
    parser.add_argument('-f', "--freeze", action="store_true", default=False, help="Freeze BERT embeddings?")
    parser.add_argument('-dr', "--dropout", type=float, default=0.1, help="Dropout amount for BERT classifier")
    parser.add_argument('-hsb', "--hidden_size_bert", type=int, default=768, help="Hidden Size for BERT classifer")
    parser.add_argument('-r', "--runs", type=int, default=1, help="Number of runs")

    args = parser.parse_args()
    average_macro_f1 = 0

    for i in range(args.runs):
      if args.model == "BERT":
          average_macro_f1 += train(args)["macro_f1"]
      elif args.model == "LSTM":
          average_macro_f1 += train_rnn(args)["macro_f1"]

    print('Average macro f1 from {} runs: {:.3f}'.format(args.runs, average_macro_f1/args.runs))