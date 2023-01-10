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
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self):
        '''
        Encoder for images of sizes 224x224 using VGG19
        architecture without the FC layers at the end.
        We use the pretrained weights without allowing
        for fine-tuning.
        '''
        super(CNNEncoder, self).__init__()
        resnet = models.resnet50(weights=[models.ResNet50_Weights])  # we use VGG19 as our encoder
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-2])
        self.conv_layers.requires_grad_(False)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)  # Flatten the input at the end to get features vector
        return x


class LSTMDecoder(nn.Module):
    def __init__(self, encoder_dim, embedding_size, hidden_size, vocab_size, device, max_seq_length=200,
                 dropout=0.1):
        '''

        :param encoder_dim: size of output encoder
        :param embedding_size: size of the vocab embedding
        :param hidden_size: hidden size used for the input features
        :param vocab_size: size of the vocabulary
        :param num_layers: number of layers in the LSTM layer
        :param max_seq_length: maximum possible sequence length
        '''
        super(LSTMDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        self.max_seq_length = max_seq_length

        # Embedding layer to map input words to word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTMCell(embedding_size, hidden_size)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.emb_features = nn.Linear(encoder_dim, embedding_size)
        self.init_weights()

    def init_weights(self):
        # We initialize all the weights
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.init_h.weight.data.uniform_(-0.1, 0.1)
        self.init_c.weight.data.uniform_(-0.1, 0.1)

    def init_h_c(self, encoder_out):
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)
        return h, c

    def forward(self, features, captions):
        batch_size = features.size(0)
        vocab_size = self.vocab_size

        emb_features = self.emb_features(features)
        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)

        h, c = self.init_h_c(features)  # (batch_size, decoder_dim)

        decode_length = captions.size(1)

        predictions = torch.zeros(batch_size, decode_length+1, vocab_size).to(self.device)

        # first pass the image through
        h, c = self.lstm(emb_features, (h, c))
        predictions[:, 0, :] = self.fc(self.dropout(h))
        # then the sentence w/o the first word
        for t in range(decode_length):
            h, c = self.lstm(embeddings[:, t, :], (h, c))  # (batch_size_t, hidden)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t + 1, :] = preds

        return predictions


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
            norm_first=True,
            num_layers=num_layers
        )
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, inputs, mask, src_mask):
        x = self.embedding(inputs)
        x = self.positional_encoding(x)
        x = self.transformer_decoder(x, src_mask, mask)
        x = self.linear(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, encoder_out, embedding_size, hidden_size, vocab_size, device,
                 num_layers=1, max_seq_length=50):
        super(EncoderDecoder, self).__init__()
        self.encoder_out = encoder_out
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(encoder_out, embedding_size, hidden_size, vocab_size, device, max_seq_length=max_seq_length)
    def forward(self, image, caption):
        features = self.encoder(image)
        output = self.decoder(features, caption)
        return output

    def init_weights(self):
        self.decoder.init_weights()

    def generate_caption(self, image, vocab):
        '''
        Create caption for image
        :param image: the image we want to create the caption of
        :param vocab: the vocabulary according to which we caption
        :return: the caption as a string
        '''
        result = []
        batch_size = len(image)
        with torch.no_grad():
            features = self.encoder(image)
            h, c = self.decoder.init_h_c(features)
            x = self.decoder.emb_features(features)

            for i in range(self.max_seq_length):
                h, c = self.decoder.lstm(x, (h, c))
                output = self.decoder.fc(h)
                prediction = output.argmax(dim=1)
                if len(result) != 0:
                    result = torch.vstack((result, prediction))
                else:
                    result = prediction
                    ''' 
                    if vocab.itos[prediction.tolist()] == "<EOS>":
                        break
                    '''
                x = self.decoder.embedding(prediction)
        predictions = []
        for i in range(batch_size):
            predictions.append([vocab.itos[idx] for idx in result[:, i].tolist()])
        return predictions