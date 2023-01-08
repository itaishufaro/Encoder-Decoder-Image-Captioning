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
    def __init__(self, encoder_out, embedding_size, hidden_size, vocab_size, num_layers=1, max_seq_length=200):
        '''

        :param encoder_out: size of output encoder
        :param embedding_size: size of the vocab embedding
        :param hidden_size: hidden size used for the input features
        :param vocab_size: size of the vocabulary
        :param num_layers: number of layers in the LSTM layer
        :param max_seq_length: maximum possible sequence length
        '''
        super(LSTMDecoder, self).__init__()
        self.encoder_out = encoder_out
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length

        # Embedding layer to map input words to word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.word_embeddings.requires_grad_(True)

        # LSTM layers
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.lstm.requires_grad_(True)

        # Linear layer to map the hidden state of the LSTM to the output vocabulary
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.linear.requires_grad_(True)

        # Linear layer to map the features to the embedded size
        self.embed_feature = nn.Linear(self.encoder_out, embedding_size)
        self.embed_feature.requires_grad_(True)

    def init_weights(self):
        # We initialize all the weights of the lstm layer using xavier_uniform
        nn.init.kaiming_normal_(self.word_embeddings.weight)
        nn.init.kaiming_normal_(self.lstm.weight_ih_l0)
        nn.init.kaiming_normal_(self.lstm.weight_hh_l0)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.kaiming_normal_(self.embed_feature.weight)

    def forward(self, features, captions):
        # Convert input captions to word embeddings
        # embeddings = self.word_embeddings(captions)

        # Concatenate features and word embeddings and pass them through the LSTM
        features_embed = self.embed_feature(features)
        embed = self.word_embeddings(captions)
        lstm_input = torch.cat((features_embed.unsqueeze(1), embed), 1)
        lstm_output, _ = self.lstm(lstm_input)

        # Use the linear layer to map the LSTM output to the output vocabulary
        outputs = self.linear(lstm_output)

        return outputs


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
    def __init__(self, encoder_out, embedding_size, hidden_size, vocab_size,
                 num_layers=1, max_seq_length=50):
        super(EncoderDecoder, self).__init__()
        self.encoder_out = encoder_out
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(encoder_out, embedding_size, hidden_size, vocab_size, num_layers, max_seq_length)

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
            x = self.encoder(image)
            x = self.decoder.embed_feature(x)
            c = None

            for i in range(self.max_seq_length):
                h, c = self.decoder.lstm(x, c)
                output = self.decoder.linear(h)
                prediction = output.argmax(dim=1)
                if len(result) != 0:
                    result = torch.vstack((result, prediction))
                else:
                    result = prediction
                    ''' 
                    if vocab.itos[prediction.tolist()] == "<EOS>":
                        break
                    '''
                x = self.decoder.word_embeddings(prediction)
        predictions = []
        for i in range(batch_size):
            predictions.append([vocab.itos[idx] for idx in result[:, i].tolist()])
        return predictions