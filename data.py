import json
import numpy as np
import pandas as pd
import cv2
import torchvision.transforms
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from PIL import Image
import spacy
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import models

spacy_eng = spacy.load("en_core_web_sm")  # load nlp model


# For prepararing the vocabulary
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(str(text))]

    def build_vocab(self, sent_list):
        freqs = {}
        idx = 4
        for sent in sent_list:
            sent = str(sent)
            for word in self.tokenize(sent):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1

                if freqs[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, sents):
        tokens = self.tokenize(sents)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokens]

    def sequence_to_text(self, seq):
        """

        :param seq: Input sequence as tensor (one sentence) 1 x NUM_WORDS
        :return: Input sequence as text (list) NUM_WORDS
        """
        tmp = seq.squeeze()
        tmp_list = [self.itos[i.item()] for i in tmp]
        SOSind = tmp_list.index("<SOS>")
        EOSind = tmp_list.index("<EOS>")
        return ' '.join(tmp_list[SOSind+5:EOSind])

    def sequences_to_texts(self, seqs):
        """

        :param seqs: Input sequences as tensor NUM_SENTENCES x NUM_WORDS
        :return: Appropriate texts as list of lists
        """

        tmp = seqs.squeeze()
        tmp_list = []
        for i in tmp:
            tmp_list.append(self.sequence_to_text(i))
        return tmp_list

# custom dataset class to get numericalized captions and images
class FlickrDataset(Dataset):
    def __init__(self, root_dir, caps, transforms=None, freq_threshold=1, im_width=128):
        '''
        :param root_dir: directory for the images
        :param caps: path to captions file
        :param transforms: transforms we wish to preform on the images
        :param freq_threshold: frequency threshold for the vocabulary
        :param im_width: new image width we work with
        '''
        self.root_dir = root_dir
        self.df = pd.read_csv(caps, delimiter='|')
        self.transforms = transforms

        self.im_width = im_width
        self.img_pts = self.df['image_name']
        self.caps = self.df[' comment']
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.caps.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        :param idx: dataset item index
        :return: image (torch.tensor) and captions (torch.tensor)
        '''
        captions = self.caps[idx]
        img_pt = self.img_pts[idx]
        img = Image.open(os.path.join(self.root_dir, img_pt)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        numberized_caps = []
        numberized_caps += [self.vocab.stoi["<SOS>"]]  # stoi string to index
        numberized_caps += self.vocab.numericalize(captions)
        numberized_caps += [self.vocab.stoi["<EOS>"]]
        return img, torch.tensor(numberized_caps)


# Padding the captions according to the largest caption in the batch
class CapCollat:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def convert_to_imshow_format(image):
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = (image - np.min(np.min(image, axis=0),axis=0))
    image = image / np.max(np.max(image,axis=0),axis=0)
    # We preform normalization on the image in order to be in range [0,1]
    return image


def show_img(img, caption):
    img = convert_to_imshow_format(img)
    plt.imshow(img)
    plt.title(caption)
    plt.show()

'''
if __name__ == "__main__":
    batch_size = 1
    root_folder = "./flickr30k/images"
    csv_file = "./flickr30k/results.csv"

    # Images normalized according to resnet 50 expectations (optional)
    transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    num_workers = 2
    batch_first = True
    pin_memory = True
    shuffle = True
    dataset = FlickrDataset(root_folder, csv_file, transforms)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    # Data Loader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            collate_fn=CapCollat(pad_seq=pad_idx, batch_first=batch_first))
    dataitr = iter(dataloader)
    batch = next(dataitr)
    images, captions = batch
    print(captions.shape)
    print(captions[0])
    # print(dataset.vocab.itos[token] for token in captions[0].tolist())
    caption_label = [dataset.vocab.itos[token] for token in captions[0].tolist()]
    eos_index = caption_label.index('<EOS>')
    caption_label = caption_label[1:eos_index]
    caption_label = ' '.join(caption_label)
    print(caption_label)
    embedding_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    encoder_out = 32768
    encodedecode = models.EncoderDecoder(encoder_out, embedding_size, hidden_size, vocab_size)
    new_word = encodedecode(images, captions)
    print(new_word.size())
    caption_label = [dataset.vocab.itos[token] for token in new_word.tolist()]
    eos_index = caption_label.index('<EOS>')
    caption_label = caption_label[1:eos_index]
    caption_label = ' '.join(caption_label)
    print(caption_label)
'''