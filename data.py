import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
from torch.utils.data import Dataset
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# For prepararing the vocabulary
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "[PAD]", 1: "[CLS]", 2: "[SEP]", 3: "[UNK]"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return self.bert_tokenizer.tokenize(text)

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
        return [self.stoi[token] if token in self.stoi else self.stoi["[UNK]"]
                for token in tokens]

    def sequence_to_text(self, seq):
        """

        :param seq: Input sequence as tensor (one sentence) 1 x NUM_WORDS
        :return: Input sequence as text (list) NUM_WORDS
        """
        tmp = seq.squeeze()
        tmp_list = [self.itos[i.item()] for i in tmp]
        SOSind = tmp_list.index('[CLS]')  # for bert
        EOSind = tmp_list.index('[SEP]')  # for bert
        return ' '.join(tmp_list[SOSind+1:EOSind])

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
        self.df = pd.read_csv(caps, delimiter=',')
        self.transforms = transforms

        self.im_width = im_width
        self.img_pts = self.df['image']
        self.caps = self.df['caption']
        self.vocab = bert_tokenizer.vocab

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
        numberized_caps.append(self.vocab["[CLS]"])  # for bert
        numberized_caps += (bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(captions)))
        numberized_caps.append(self.vocab["[SEP]"])  # for bert
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

def train_valid_test_split(df, train_percent=.8, valid_percent=.1, seed=0):
    '''
    :param df: dataframe to split
    :param train_percent: percentage of the data to be used for training
    :param valid_percent: percentage of the data to be used for validation
    :param seed: random seed
    '''
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    valid_end = int(valid_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    valid = df.iloc[perm[train_end:valid_end]]
    test = df.iloc[perm[valid_end:]]
    return train, valid, test


def convert_to_imshow_format(image):
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = (image - np.min(np.min(image, axis=0),axis=0))
    image = image / np.max(np.max(image,axis=0),axis=0)
    # We preform normalization on the image in order to be in range [0,1]
    return image


def show_img(img, caption):
    plt.axis('off')
    img = convert_to_imshow_format(img)
    plt.imshow(img)
    plt.title(caption)
    plt.show()