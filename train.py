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


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def train_one_epoch(model, dataloader, optimizer, criterion, device, augmentations=None):
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
    i = 1
    tot_loss = 0
    len_dataset = len(dataloader)
    for images, captions in iter(dataloader):
        images, captions = images.to(device), captions.to(device)
        if not augmentations is None:
            images = augmentations(images)
        captions = captions.transpose(0, 1)
        optimizer.zero_grad()
        output = model(images, captions[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[2]), captions.reshape(-1))
        # loss = criterion(output.reshape(-1, output.shape[2]), captions.reshape(-1))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        '''if i % 100 == 0:
            # eval_score = evaluate(model, dataloader, vocab)
            # print("For iteration " + str(i) + " the loss is : " + str(tot_loss))
            # tot_loss = 0
            image, caption = next(singledataloader)
            caption = bert_tokenizer.convert_ids_to_tokens(caption.squeeze().tolist())
            caption = bert_tokenizer.convert_tokens_to_string(caption)
            print("True caption: " + caption)
            model.eval()
            image = image.to(device)
            pred, _ = generate_caption(model, image, vocab, device)
            print("Prediction: " + pred)
            model.train()
            # print("For iteration " + str(i) + " the belu is : " + str(eval_score))'''
        i = i + 1
        tot_loss += loss.item()
        torch.cuda.empty_cache()
    return tot_loss/len_dataset

def train_epochs(num_epochs, model, dataloader, optimizer, criterion, device, validloader, vocab,
                 augmentations=None):
    loss_points = []
    valid_points = []
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch + 1))
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device, augmentations)
        print("Train Loss: " + str(loss))
        loss_points.append(loss)
        eval_score = evaluate(model, validloader, vocab, device, criterion)
        print("Eval Score: " + str(eval_score))
        valid_points.append(eval_score)
    final_loss = loss_points[-1]
    final_score = valid_points[-1]
    return loss_points, valid_points, final_loss, final_score


def generate_caption(model, image, vocab, device, max_len=50, temp=0.1, batch_size=128):
    '''

    :param model: The model we use for image captioning
    :param image: The image we wish to predict the caption
    :param vocab: The vocabulary we use
    :param max_len: The maximum length of the caption
    :param temp: The temperature we use for the softmax
    :return: The predicted caption
    '''
    model.eval()
    with torch.no_grad():
        # encoder_out = model.CNNEncoder(image)
        encoder_out = model.encoder(image)
        # inputs = torch.tensor(vocab.stoi["<SOS>"]).unsqueeze(0)
        inputs = torch.tensor(vocab["[CLS]"]).unsqueeze(0).to(device)
        predicted_caption = inputs
        predicted_caption = predicted_caption.repeat(batch_size, 1)
        for i in range(max_len):
            # output = model.TransformerDecoder(encoder_out, predicted_caption.unsqueeze(0))
            output = model.decoder(encoder_out, predicted_caption)
            output = output.squeeze()
            '''word_weights = output[1,:-1,:].squeeze().div(temp).exp()
            word_idx = torch.multinomial(word_weights, 1)[0]'''
            word_idx = output[:, -1].argmax(1)
            predicted = word_idx
            if len(predicted.size()) == 0:
                predicted = predicted.unsqueeze(0)
            predicted_caption = torch.cat((predicted_caption, predicted.unsqueeze(1)), 1)
    torch.cuda.empty_cache()
    # return [vocab.itos[idx] for idx in predicted_caption.tolist()]
    # tmp = [bert_tokenizer._convert_id_to_token(idx) for idx in predicted_caption.tolist()]
    return predicted_caption[:, :-1]


def evaluate(model, validloader, vocab, device, criterion):
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
        prob_vector = model(images, pred_caption[:, :-1])
        tmp = criterion(prob_vector.reshape(-1, prob_vector.shape[2]), captions.reshape(-1))
        eval_loss += tmp.item()
    return eval_loss/dataset_size


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
    embedding_size = 768  # 768 default for BERT
    hidden_size = 128
    vocab_size = len(vocabulary)
    '''
    Next we define our model and hyper-parameters
    '''
    encoder_out = 2048
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # model = EncoderTransformerDecoder(encoder_out, embedding_size, vocab_size, hidden_size , num_layers=1)
    # model = EncoderTransformerDecoder(encoder_out, embedding_size, vocab_size, num_hiddens=embedding_size, num_layers=1)
    model = LSTMDecoderEncoderBERT(embedding_size, hidden_size, vocab_size, num_layers=2)
    # model.init_weights()
    '''
    Hyperparameters
    '''
    batch_size = 128
    print(device)
    num_workers = 2
    batch_first = True
    pin_memory = True
    shuffle = True
    # pad_idx = dataset.vocab.stoi["<PAD>"]  # for spacy
    pad_idx = dataset.vocab["[PAD]"]  # for bert
    dataset_size = len(dataset)
    trainloader = DataLoader(traindata,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            collate_fn=data.CapCollat(pad_idx=pad_idx))
    validloader = DataLoader(validdata,
                             batch_size=batch_size,
                             pin_memory=pin_memory,
                             num_workers=num_workers,
                             shuffle=shuffle,
                             collate_fn=data.CapCollat(pad_idx=pad_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.95
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)  #
    model.to(device)
    num_epochs = 50
    singleimageloader = DataLoader(dataset,
                                   batch_size=1,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   shuffle=shuffle)
    aug_list = AugmentationSequential(
        K.RandomAffine(5, [0.05, 0.05], [0.95, 1.05], p=.1),
        # Preforms random affine transformation (rotation, translation, scale).
        # We limited to 5 degrees of rotation, small translation and small scale
        K.RandomPerspective(0.1, p=.1),
        # Changes perspective of image.
        # K.RandomGaussianNoise(mean=0., std=0.005, p=0.5),
        # Adds random gaussian noise with mean 0 and std of 0.1
        same_on_batch=False
    )
    train_epochs(num_epochs, model, trainloader, optimizer, criterion, device, validloader, vocabulary,
                 augmentations=None)
    model.eval()
    torch.save(model, 'LSTM_BERT.pth')

    # pretrained word embedding (Bert) - done
    # perflix - check gits in mail
    # kornia augmantations - done
    # check HW to see transformer image generation - done
    # optuna for hyperparameter tuning
    # train test split
