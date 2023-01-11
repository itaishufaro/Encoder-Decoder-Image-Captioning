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
        Encoder for images of using resnet50
        architecture without the FC layers at the end.
        We use the pretrained weights without allowing
        for fine-tuning.
        '''
        super(CNNEncoder, self).__init__()
        resnet = models.resnet50(weights=[models.ResNet50_Weights])  # we use resnet50 as our encoder
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-2])
        self.avg_pool = nn.AvgPool2d(2)
        self.conv_layers.requires_grad_(False)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
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
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1)
        x = x / torch.pow(10_000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        tmp = x + self.P[:, :x.shape[1], :].to(x.device)
        return self.dropout(tmp)


class TranDecoder(nn.Module):
    def __init__(self, encoder_out_dim, embedding_size, vocab_size, num_hiddens,
                 num_layers, num_heads=2, dropout=0.1):
        '''
        :param encoder_out_dim: Output dimension of encoder
        :param output_dim: Output dimension of decoder (vocab size)
        :param num_layers: Number of transformer layers
        :param vocab_size: Vocabulary size
        :param num_heads: Number of heads in MHA
        :param hidden_dim: Hidden dimension used for embedding of the features
                            given by the decoder
        :param dropout: dropout probability for features
        '''
        super(TranDecoder, self).__init__()
        # self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.encoder_out_dim = encoder_out_dim
        self.embedding_size = embedding_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding = PositionalEncoding(num_hiddens, dropout)
        self.TransformerLayer = nn.TransformerDecoderLayer(embedding_size, num_heads, dropout=dropout,
                                                      activation="gelu", batch_first=True)
        self.Transformer = nn.TransformerDecoder(self.TransformerLayer, self.num_layers)
        self.fc = nn.Linear(embedding_size, vocab_size)
        self.feature_linear = nn.Linear(encoder_out_dim, embedding_size)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.feature_linear.weight)
        for p in self.Transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, captions):
        """

        :param features: Input features
        :param captions: Input target captions
        :return: Transofrmer sequential output
        """
        emb_features = self.feature_linear(features)
        embed = self.word_embedding(captions)
        positional_embed = self.positional_embedding(embed)
        out = self.Transformer(tgt=positional_embed, memory=emb_features.unsqueeze(1))
        out = self.fc(out)
        return out

    # def generate_caption(self, features):


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
        predictions = [item for sublist in predictions for item in sublist]
        SOS_ind = 0
        EOS_ind = len(predictions)
        if "<SOS>" in predictions:
            SOS_ind = predictions.index("<SOS>") + 1
        if "<EOS>" in predictions:
            EOS_ind = predictions.index("<EOS>")
        return predictions[SOS_ind:EOS_ind]


class EncoderTransformerDecoder(nn.Module):
    def __init__(self, encoder_out_dim, embedding_size, vocab_size, num_hiddens,
                 num_layers, num_heads=2, dropout=0.1):

        super(EncoderTransformerDecoder, self).__init__()
        self.CNNEncoder = CNNEncoder()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.encoder_out_dim = encoder_out_dim
        self.embedding_size = embedding_size
        self.TransformerDecoder = TranDecoder(encoder_out_dim, embedding_size, vocab_size, num_hiddens, num_layers)

    def forward(self, images, captions):
        enc_out = self.CNNEncoder(images)
        dec_out = self.TransformerDecoder(enc_out, captions)
        return dec_out





