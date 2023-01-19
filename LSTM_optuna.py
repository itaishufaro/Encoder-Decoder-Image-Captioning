from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import data
from data import FlickrDataset, train_valid_test_split
import torch
from torch import nn
from models import EncoderTransformerDecoder, EncoderDecoder, LSTMDecoderEncoderBERT
import kornia
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
from transformers import BertTokenizer
import optuna
import train
from train import train_epochs


def objective(trial, device, train_dataloader, valid_dataloader, vocab, bert_tokenizer, aug_list, num_epochs=10,
              embedding_size=768):
    # We optimize the number of layers, hidden units and dropout in each layer.
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_int("hidden_size", 256, 512)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    model = LSTMDecoderEncoderBERT(hidden_size=hidden_size, num_layers=num_layers, vocab_size=len(vocab),
                                   embed_size=embedding_size)
    model = model.to(device)
    pad_idx = bert_tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    _, _, final_loss, final_score = train_epochs(num_epochs=num_epochs, model=model, dataloader=train_dataloader,
                                                 optimizer=optimizer, criterion=criterion, device=device,
                                                 validloader=valid_dataloader, vocab=vocab, augmentations=aug_list)
    return final_score



if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    '''
    First, create the flickr dataset
    '''
    transforms = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    root_folder = "./flickr8k/images"
    csv_file = "./flickr8k/captions.txt"
    dataset = FlickrDataset(root_folder, csv_file, transforms)
    traindata, validdata, testdata = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    # Important variables for later on
    vocabulary = dataset.vocab
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Create the dataloaders
    pin_memory = True
    num_workers = 2
    shuffle = True
    pad_idx = dataset.vocab["[PAD]"]
    train_dataloader = DataLoader(dataset,
                                   batch_size=128,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   shuffle=shuffle,
                                  collate_fn=data.CapCollat(pad_idx=pad_idx))
    valid_dataloader = DataLoader(dataset,
                                   batch_size=128,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   shuffle=shuffle,
                                  collate_fn=data.CapCollat(pad_idx=pad_idx))
    # test_dataloader = DataLoader(testdata, batch_size=32, shuffle=True, num_workers=1)
    # Create the augmentations
    aug_list = AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomAffine(5, [0.05, 0.05], [0.95, 1.05], p=.1),
        K.RandomPerspective(0.1, p=.1),
        same_on_batch=False)
    # Create the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the study
    study = optuna.create_study(study_name="LSTM_optuna",
        storage='sqlite:///LSTM_optuna.db',
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial, device, train_dataloader, valid_dataloader, vocabulary,
                                           bert_tokenizer, aug_list), n_trials=1)
