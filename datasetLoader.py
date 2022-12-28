# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
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

spacy_eng = spacy.load("en_core_web_sm")


# For prepararing the vocabulary

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

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


# custom dataset class to get numericalized captions and images
class FlickrDataset(Dataset):
    def __init__(self, root_dir, caps, transforms=None, freq_threshold=5, im_width=128):
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
        captions = self.caps[idx]
        img_pt = self.img_pts[idx]
        img = Image.open(os.path.join(self.root_dir, img_pt)).convert('RGB')
        w,h = img.size
        new_h = int(h / w * self.im_width)
        newsize = (self.im_width, new_h)
        img = img.resize(newsize)
        if self.transforms is not None:
            img = self.transforms(img)

        numberized_caps = []
        numberized_caps += [self.vocab.stoi["<SOS>"]]  # stoi string to index
        numberized_caps += self.vocab.numericalize(captions)
        numberized_caps += [self.vocab.stoi["<EOS>"]]
        return img, torch.tensor(numberized_caps)


# Padding the captions according to the largest caption in the batch
class CapCollat:
    def __init__(self, pad_seq, batch_first=False):
        self.pad_seq = pad_seq
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [itm[0].unsqueeze(0) for itm in batch]
        imgs = torch.cat(imgs, dim=0)

        target_caps = [itm[1] for itm in batch]
        target_caps = pad_sequence(target_caps, batch_first=self.batch_first,
                                   padding_value=self.pad_seq)
        return imgs, target_caps
def convert_to_imshow_format(image):
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = (image - np.min(np.min(image, axis=0),axis=0))
    image = image / np.max(np.max(image,axis=0),axis=0)
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image
def show_img(img, caption):
    img = convert_to_imshow_format(img)
    plt.imshow(img)
    plt.title(caption)
    plt.show()
if __name__ == "__main__":
    batch_size = 3
    root_folder = "./flickr30k/images"
    csv_file = "./flickr30k/results.csv"

    # Images normalized according to resnet 50 expectations (optional)
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    num_workers = 2
    batch_first = True
    pin_memory = True
    shuffle = True
    dataset = FlickrDataset(root_folder, csv_file, transforms, im_width=128)
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
    for i in range(batch_size):
        img, cap = images[i], captions[i]
        caption_label = [dataset.vocab.itos[token] for token in cap.tolist()]
        eos_index = caption_label.index('<EOS>')
        caption_label = caption_label[1:eos_index]
        caption_label = ' '.join(caption_label)
        show_img(img, caption_label)
        plt.show()