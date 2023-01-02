import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
import torchvision.transforms as T
import torchvision
import data
from data import FlickrDataset
import math


class CNNEncoder(nn.Module):
    def __init__(self):
        '''
        Encoder for images of sizes 224x224 using VGG19
        architecture without the FC layers at the end.
        We use the pretrained weights without allowing
        for fine-tuning.
        '''
        super(CNNEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True)  # we use VGG19 as our encoder
        # We also freeze the weights
        vgg.requires_grad_ = False
        self.vgg = vgg  # extract only features layer

    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.shape[0], -1)  # Flatten the input at the end to get features vector
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        '''
        :param d_model: dimension
        :param dropout: dropout probability
        :param max_len: maximal word length
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, encoder_out_dim, output_dim, num_layers, num_heads,
                 hidden_dim, dropout=0.1):
        '''
        :param encoder_out_dim: Output dimension of encoder
        :param output_dim: Output dimension of decoder (vocab size)
        :param num_layers: Number of transformer layers
        :param num_heads: Number of heads in MHA
        :param hidden_dim: Hidden dimension used for embedding of the features
                            given by the decoder
        :param dropout: dropout probability for features
        '''
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = nn.Linear(in_features=encoder_out_dim, out_features=hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout),
            num_layers=num_layers
        )
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, inputs, mask, src_mask):
        x = self.embedding(inputs)
        x = self.positional_encoding(x)
        x = self.transformer_decoder(x, src_mask, mask)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    model = CNNEncoder()
    batch_size = 1
    root_folder = "./flickr30k/images"
    csv_file = "./flickr30k/results.csv"
    # Check if a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")  # use GPU
    else:
        device = torch.device("cpu")  # use CPU
    model.to(device)
    # Images normalized according to VGG19 expectations (optional)
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
    dataset = FlickrDataset(root_folder, csv_file, transforms)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    # Data Loader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            collate_fn=data.CapCollat(pad_seq=pad_idx, batch_first=batch_first))
    dataitr = iter(dataloader)
    batch = next(dataitr)
    images, captions = batch
    img_features = model(images.to(device))
    vocabsize = len(dataset.vocab.itos)


