import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        '''
        Encoder for images of using resnet50
        architecture without the FC layers at the end.
        We use the pretrained weights without allowing
        for fine-tuning.
        The only trainable layer is the linear layer at the end
        which is used to increase expressiveness of the features,
        and to reduce the dimensionality of the features to embedding size.
        '''
        super(CNNEncoder, self).__init__()
        resnet = models.resnet50(weights=[models.ResNet50_Weights])  # we use resnet50 as our encoder
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])
        self.conv_layers.requires_grad_(False)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.conv_layers(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class LSTMDecoderBERT(nn.Module):
    """LSTM Decoder with BERT Embedding"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(LSTMDecoderBERT, self).__init__()
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        weight = bert_model.embeddings.word_embeddings.weight
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.prembed = nn.Embedding.from_pretrained(weight)
        self.prembed.requires_grad_(False)
        self.init_weights()

    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        with torch.no_grad():
            embeddings = self.prembed(captions).float()
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.kaiming_normal_(self.lstm.weight_ih_l0)
        nn.init.kaiming_normal_(self.lstm.weight_hh_l0)
        self.linear.bias.data.fill_(0)


class LSTMDecoderEncoderBERT(nn.Module):
    """Full encoder decoder model with LSTM decoder"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(LSTMDecoderEncoderBERT, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = LSTMDecoderBERT(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)

    def forward(self, images, captions):
        """Decode image feature vectors and generates captions."""
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


class TranDecoder(nn.Module):
    """Transformer Decoder with BERT Embedding"""
    def __init__(self, embed_size, vocab_size, num_layers, n_head, max_seq_length=20,
                 norm_first=False):
        super(TranDecoder, self).__init__()
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=n_head, batch_first=True,
                                                                  norm_first=norm_first)
        self.transformer = nn.TransformerDecoder(self.TransformerDecoderLayer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.max_seg_length = max_seq_length
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.prembed = bert_model.embeddings
        self.prembed.requires_grad_(False)
        self.init_weights()


    def init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.kaiming_normal_(self.TransformerDecoderLayer.self_attn.in_proj_weight)
        nn.init.kaiming_normal_(self.TransformerDecoderLayer.multihead_attn.in_proj_weight)
        self.linear.bias.data.fill_(0)

    def generate_mask(self, size, decoder_inp):
        """Generate a square mask for the sequence. The masked positions are filled with float(‘-inf’)."""
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(
            decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(
            decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, features, decoder_inp):
        """Get feature vectors and captions, and return the output of the transformer decoder."""
        decoder_inp_embed = self.prembed(decoder_inp)
        decoder_input_mask, _, decoder_input_pad_mask_bool = self.generate_mask(
            decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask.to(device)
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool.to(device)
        decoder_output = self.transformer(tgt=decoder_inp_embed, memory=features.unsqueeze(1),
                                          tgt_mask=decoder_input_mask,
                                          tgt_key_padding_mask=decoder_input_pad_mask_bool)
        outputs = self.linear(decoder_output)
        return outputs


class TransformerEncoderDecoder(nn.Module):
    """Full encoder decoder model with Transformer decoder"""
    def __init__(self, embed_size, vocab_size, num_layers, n_head, max_seq_length=20,
                 norm_first=False):
        super(TransformerEncoderDecoder, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = TranDecoder(embed_size, vocab_size, num_layers, n_head, max_seq_length,
                                   norm_first=norm_first)
        self.decoder.init_weights()

    def forward(self, images, captions):
        """Decode image feature vectors and generates captions."""
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs