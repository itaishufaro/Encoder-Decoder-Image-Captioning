from torch.utils.data import DataLoader
import torchvision.transforms as T
import data
from data import FlickrDataset
import torch
from torch import nn
from models import TransformerEncoderDecoder
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
from transformers import BertTokenizer
import optuna
from train import train_epochs
import pytorch_warmup as warmup


def objective(trial, device, train_dataloader, valid_dataloader, vocab, bert_tokenizer, aug_list, num_epochs=8,
              embedding_size=768):
    # We optimize the number of layers, hidden units and dropout in each layer.
    lr = trial.suggest_float("lr", 1e-6, 1e-3)
    weight_decay = 0
    gamma = 0.99
    num_layers = 1
    n_head = 2
    norm_first = trial.suggest_categorical("norm_first", [True, False])
    scheduler = trial.suggest_categorical("scheduler", ['ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    optimizer = trial.suggest_categorical("optimizer", ['RAdam', 'SGD'])
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)
    model = TransformerEncoderDecoder(num_layers=num_layers, vocab_size=len(vocab),
                                      embed_size=embedding_size, n_head=n_head, norm_first=norm_first)
    model = model.to(device)
    pad_idx = bert_tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    if optimizer == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_steps)
    if scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        scheduler = None
    _, _, temp_loss, temp_valid = train_epochs(num_epochs=3, model=model, dataloader=train_dataloader,
                                                 optimizer=optimizer, criterion=criterion, device=device,
                                                 validloader=valid_dataloader, vocab=vocab, augmentations=aug_list)
    if temp_loss > 10.0:
        return temp_loss, temp_valid

    _, _, final_loss, final_valid = train_epochs(num_epochs=num_epochs-3, model=model, dataloader=train_dataloader,
                                                 optimizer=optimizer, criterion=criterion, device=device,
                                                 validloader=valid_dataloader, vocab=vocab, augmentations=aug_list)
    print('-' * 10)
    print(f"Final loss: {final_loss}, Final Validation: {final_valid}")
    print('-' * 10)
    torch.cuda.empty_cache()
    return final_loss, final_valid



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
    root_folder = "./flickr8k/Images"
    csv_file = "./flickr8k/captions.txt"
    dataset = FlickrDataset(root_folder, csv_file, transforms)
    # set the seed for reproducibility
    torch.manual_seed(123)
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
                                   batch_size=256,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   shuffle=shuffle,
                                  collate_fn=data.CapCollat(pad_idx=pad_idx))
    valid_dataloader = DataLoader(dataset,
                                   batch_size=256,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   shuffle=shuffle,
                                  collate_fn=data.CapCollat(pad_idx=pad_idx))
    # Create the augmentations
    aug_list = AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomAffine(5, [0.05, 0.05], [0.95, 1.05], p=.1),
        K.RandomPerspective(0.1, p=.1),
        same_on_batch=False)
    # Create the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the study
    study = optuna.create_study(study_name="Transformer_optuna_INIT",
        storage='sqlite:///Transformer_optuna_INIT.db',
        load_if_exists=True,
        directions=["minimize", "maximize"])
    study.optimize(lambda trial: objective(trial, device, train_dataloader, valid_dataloader, vocabulary,
                                           bert_tokenizer, aug_list), n_trials=30)
