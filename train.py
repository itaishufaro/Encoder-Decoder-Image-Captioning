import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import data
from data import FlickrDataset, train_valid_test_split
import torch
from torch import nn
from models import LSTMDecoderEncoderBERT, TransformerEncoderDecoder
import kornia
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
from transformers import BertTokenizer
import pytorch_warmup as warmup
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import matplotlib.pyplot as plt


def train_one_epoch(model, dataloader, optimizer, criterion, device, augmentations=None,
                    model_class = 'Transformer'):
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
    scaler = torch.cuda.amp.GradScaler()
    i = 1
    tot_loss = 0
    len_dataset = len(dataloader)
    for images, captions in iter(dataloader):
        images, captions = images.to(device), captions.to(device)
        if not augmentations is None:
            images = augmentations(images)
        captions = captions.transpose(0, 1)
        #
        with torch.cuda.amp.autocast():
            outputs = model(images, captions[:, :-1])
            if model_class == 'Transformer':
                loss = criterion(outputs.view(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
            else:
                loss = criterion(outputs.view(-1, outputs.shape[-1]), captions.reshape(-1))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        i = i + 1
        tot_loss += loss.item()
        torch.cuda.empty_cache()
    return tot_loss/len_dataset

def train_epochs(num_epochs, model, dataloader, optimizer, criterion, device, validloader, vocab,
                 augmentations=None, model_class='Transformer', start_epoch=0):
    '''
    Training function for our model, which trains the function for multiple epochs and saves checkpoints
    :param num_epochs: The number of epochs we wish to train for
    :param model: The model we wish to train according to
    :param dataloader: The dataloader we use to load the training data
    :param optimizer: The optimizer we use to change the weights
    :param criterion: The criterion according to which we optimize
    :param device: The device we use to run the calculations
    :param validloader: The dataloader we use to load the validation data
    :param vocab: The vocabulary we use
    :param augmentations: The augmentations we use
    :param model_class: The class of the model we use (Transformer or LSTM)
    :param start_epoch: The epoch we start training from
    :return: The loss after training for each epoch, the test score after training for each epoch
    '''

    loss_points = []
    valid_points = []
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch + start_epoch + 1))
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device, augmentations,
                               model_class)
        eval_score = evaluate(model, validloader, vocab, device, model_class)
        loss_points.append(loss)
        print("Train Loss: " + str(loss))
        print("Eval Score: " + str(eval_score))
        valid_points.append(eval_score)
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), model_class + str(epoch+start_epoch+1) + ".pt")
            df = pd.DataFrame({'train_loss': loss_points, 'test_belu': valid_points})
            df.to_csv("R"+model_class + str(epoch+start_epoch+1) + ".csv")
            print("Model saved")
    final_loss = loss_points[-1]
    final_score = valid_points[-1]
    return loss_points, valid_points, final_loss, final_score


def generate_caption(model, image, vocab, device, max_len=50, temp=0, batch_size=128):
    '''

    :param model: The model we use for image captioning
    :param image: The image we wish to predict the caption
    :param vocab: The vocabulary we use
    :param max_len: The maximum length of the caption
    :param temp: The temperature we use for the softmax
    :return: The predicted caption (tokenized)
    '''
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(image)
        # inputs = torch.tensor(vocab.stoi["<SOS>"]).unsqueeze(0)
        inputs = torch.tensor(vocab["[CLS]"]).unsqueeze(0).to(device)
        predicted_caption = inputs.repeat(batch_size, 1)
        for i in range(max_len):
            output = model.decoder(encoder_out, predicted_caption)
            if temp == 0:
                word_idx = output[:, -1].argmax(dim=1)
            else:
                word_weights = output[:, -1, :].squeeze().div(temp).exp()
                word_idx = torch.multinomial(word_weights, 1)
            predicted = word_idx
            if len(predicted.size()) == 1:
                predicted = predicted.unsqueeze(0)
            predicted_caption = torch.cat((predicted_caption, predicted.T), 1)
    torch.cuda.empty_cache()
    return predicted_caption[:, :-1]


def evaluate(model, validloader, vocab, device, model_class='Transformer'):
    '''

    :param model: The model we use for image captioning
    :param validloader: The images we wish to predict the caption
    :param vocab: The vocabulary we use
    :return: The mean cross entropy loss
    '''
    model.eval()
    dataset_size = len(validloader)
    eval_loss = 0
    for images, captions in iter(validloader):
        images, captions = images.to(device), captions.to(device)
        captions = captions.transpose(0, 1)
        max_len = captions.shape[1]
        batch_size = images.shape[0]
        pred_caption = generate_caption(model, images, vocab, device, max_len, batch_size=batch_size)
        if model_class == 'Transformer':
            cap_compare = captions[:, 1:]
        else:
            cap_compare = captions
        eval_loss += corpus_bleu([[cap.tolist()] for cap in cap_compare], pred_caption.tolist(), weights=(1, 0, 0, 0),
                                 smoothing_function=SmoothingFunction().method1)
    return eval_loss/dataset_size


if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    '''
    First, create the flickr dataset
    '''
    transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    root_folder = "./flickr8k/Images"
    csv_file = "./flickr8k/captions.txt"
    dataset = FlickrDataset(root_folder, csv_file, transforms)
    # Preform split to train and test (we set manual seed so the split will be the same every time)
    torch.manual_seed(123)
    traindata, testdata = torch.utils.data.random_split(dataset, [0.9, 0.1])
    # Important variables for later on
    vocabulary = dataset.vocab
    '''
    Hyperparameters for the architecture 
    '''
    embedding_size = 768  # 768 default for BERT
    num_epochs = 100
    '''
    Next we define our model and hyper-parameters
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    Parameters for dataloader
    '''
    vocab_size = len(vocabulary)
    batch_size = 128
    num_workers = 2
    batch_first = True
    pin_memory = True
    shuffle = True
    # pad_idx = dataset.vocab.stoi["<PAD>"]  # for spacy
    pad_idx = dataset.vocab["[PAD]"]  # for bert
    dataset_size = len(dataset)
    '''
    Create the dataloaders
    '''
    trainloader_full = DataLoader(traindata,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            collate_fn=data.CapCollat(pad_idx=pad_idx))
    testloader = DataLoader(testdata,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            collate_fn=data.CapCollat(pad_idx=pad_idx))
    '''
    Augmentation list
    '''
    aug_list = AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        # Horizontal Flip should not change the caption (at least not in a significant way)
        K.RandomAffine(5, [0.05, 0.05], [0.95, 1.05], p=.5),
        # Preforms random affine transformation (rotation, translation, scale).
        # We limited to 5 degrees of rotation, small translation and small scale
        K.RandomPerspective(0.1, p=.5),
        # Changes perspective of image.
        same_on_batch=False
    )
    '''
    Optimal Hyperparameters for Transformer,
    found with optuna
    '''
    lr = 0.000952188
    hidden_size = 256
    weight_decay = 0
    gamma = 0.99
    num_layers = 2
    n_head = 3
    beta1 = 0.9
    beta2 = 0.99
    eps = 1e-8
    norm_first = True
    torch.manual_seed(123)
    model = TransformerEncoderDecoder(num_layers=num_layers, vocab_size=len(vocabulary),
                                      embed_size=768, n_head=n_head, norm_first=norm_first)
    model = model.to(device)

    pad_idx = bert_tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(beta1, beta2), eps=eps)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    warmup_scheduler = warmup.RAdamWarmup(optimizer)
    loss_points, test_points, final_loss, final_test = train_epochs(num_epochs=num_epochs, model=model,
                                                                    dataloader=trainloader_full,
                                                                    optimizer=optimizer, criterion=criterion,
                                                                    device=device, validloader=testloader,
                                                                    vocab=vocabulary, augmentations=aug_list,
                                                                    model_class='Transformer', start_epoch=0)
    model.eval()
    filename = 'Transformer_final' # Enter filename here
    torch.save(model, filename + '.pth')
    plt.figure()
    plt.plot(loss_points, label="Train loss")
    plt.plot(test_points, label="Test Score")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    print("Final train loss: ", final_loss)
    print("Final test loss: ", final_test)
    df = pd.DataFrame({"Train loss": loss_points, "Test loss": test_points})
    df.to_csv(filename + '.csv')
