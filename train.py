import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torchtext.vocab import GloVe
import torchvision.transforms as T
import torchvision
import data
from data import FlickrDataset
import math
import numpy as np
import pandas as pd
import cv2
import torchvision.transforms
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import spacy
import os
import models
from models import EncoderDecoder
from nltk.translate.bleu_score import sentence_bleu

def train(model, dataloader, optimizer, criterion, device, vocab):
    '''
    Training function for our model. Each training loop corresponds to one epoch

    :param model: The model we wish to train according to
    :param dataloader: The dataloader we use to load the training data
    :param optimizer: The optimizer we use to change the weights
    :param criterion: The criterion according to which we optimize
    :param device: The device we use to run the calculations
    :return: The loss after training on one batch
    '''
    model.train()
    i = 0
    for images, captions in iter(dataloader):
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        output = model(images, captions[:,:-1])
        loss = criterion(output.reshape(-1, output.shape[2]), captions.reshape(-1))
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            # eval_score = evaluate(model, dataloader, vocab)
            print("For iteration " + str(i) + " the loss is : " + str(loss.item()))
            # print("For iteration " + str(i) + " the belu is : " + str(eval_score))
        i = i + 1
    return loss.item()


def batch_belu(captions_list, predictions_list):
    j = len(captions_list)
    tmp = 0
    for i in range(j):
        tmp = tmp + sentence_bleu(captions_list[i], predictions_list[i])
    return tmp/j


def evaluate(model, val_dataset, vocab):
    '''

    :param model: The model we use for image captioning
    :param val_dataset: The images we wish to predict the caption
    :param vocab: The vocabulary we use
    :return: The mean belu score
    '''
    model.eval()
    belu_score = 0
    dataset_size = len(val_dataset)
    for images, captions in iter(val_dataset):
        batch_size = len(images)
        images, captions = images.to(device), captions.to(device)
        predicted_captions = model.generate_caption(images, vocab)
        captions_list = []
        for i in range(batch_size):
            captions_list.append([vocab.itos[idx] for idx in captions[i, :].tolist()])
        belu_score = belu_score + sentence_bleu(captions_list, predicted_captions)
    return belu_score/dataset_size


if __name__ == '__main__':
    '''
    First, create the flickr dataset
    '''
    transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    root_folder = "./flickr30k/images"
    csv_file = "./flickr30k/results.csv"
    dataset = FlickrDataset(root_folder, csv_file, transforms)
    # Important variables for later on
    vocabulary = dataset.vocab
    embedding_size = 256
    hidden_size = 256
    vocab_size = len(vocabulary)
    '''
    Next we define our model and hyper-parameters
    '''
    encoder_out = 32768
    model = EncoderDecoder(encoder_out, embedding_size, hidden_size, vocab_size)
    model.init_weights()
    '''
    Hyperparameters
    '''
    batch_size = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    num_workers = 2
    batch_first = True
    pin_memory = True
    shuffle = True
    pad_idx = dataset.vocab.stoi["<PAD>"]
    dataset_size = len(dataset)
    trainloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            collate_fn=data.CapCollat(pad_seq=pad_idx, batch_first=batch_first))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    model.to(device)
    num_epochs = 3
    for i in range(num_epochs):
        trainloss = train(model, trainloader, optimizer, criterion, device, vocabulary)
    torch.save(model.state_dict(), 'LSTMdecoder.pt')
    singleimageloader = DataLoader(dataset,
                            batch_size=1,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            collate_fn=data.CapCollat(pad_seq=pad_idx, batch_first=batch_first))
    itr = iter(singleimageloader)
    image, caption = next(itr)
    caption_label = [dataset.vocab.itos[token] for token in caption.tolist()]
    eos_index = caption_label.index('<EOS>')
    caption_label = caption_label[1:eos_index]
    caption_label = ' '.join(caption_label)
    print("True Caption Is: " + caption_label)
    predicted_caption = model.generate_caption(image, vocabulary)
    eos_index = predicted_caption.index('<EOS>')
    caption_label = predicted_caption[1:eos_index]
    caption_label = ' '.join(caption_label)
    print("Generated Caption Is: " + caption_label)
