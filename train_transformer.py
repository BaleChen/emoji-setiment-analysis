import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk.tokenize as tk
import gensim as gsm
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import random
import utils

# Setting random seed for result reproduction
SEED = 2022

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class TweetDataset(Dataset):

    def __init__(self, tweets, targets, tokenizer, max_len):
        self.tweets = tweets
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweets = utils.remove_emojis(str(self.tweets[item]))
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
          tweets,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )

        return {
          'tweet_text': tweets,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.float32)
        }


def create_data_loader(X, y, tokenizer, max_len, batch_size):
    ds = TweetDataset(
    tweets=X,
    targets=y,
    tokenizer=tokenizer,
    max_len=max_len
    )

    return DataLoader(
    ds,
    batch_size=batch_size)


class BERT_BiLSTM_FFF_Cls(nn.Module):

    def __init__(self, bert, hidden_dim, bidirectional):
        super(BERT_BiLSTM_FFF_Cls, self).__init__()
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.hidden_dim = hidden_dim
        self.bert = bert
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=2,
                            bidirectional=self.bidirectional,
                            batch_first=True)

        self.drop = nn.Dropout(p=0.2)

        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            embeddings = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]

        _, (out, _) = self.lstm(embeddings)

        # hidden = [n layers * n directions, batch size, emb dim]
        out = torch.cat((out[-2, :, :], out[-1, :, :]), dim=1)
        out = self.drop(out)

        return self.out(out)

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )
#         print(outputs.dtype)
        _,preds = torch.max(outputs, dim = 1)
        loss = loss_fn(outputs, targets)
#         prediction_error += torch.sum(torch.abs(targets - outputs))
        correct_predictions += torch.sum(preds == torch.max(targets, dim = 1)[1])
#         print(f'Iteration loss: {loss.item()}')
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)
    # return np.mean(losses), np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            #             prediction_error += torch.sum(torch.abs(targets - outputs))
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == torch.max(targets, dim = 1)[1])
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)
    # return nn.functional.cosine_similarity(output_all, target_all, dim=0), np.mean(losses)

if __name__ == "__main__":
    # Reading data and pretrained embeddings
    data = pd.read_excel('Data/emoji2vec_data/emoji2vec_train.xlsx')[['content', 'label']]
    test = pd.read_excel('Data/emoji2vec_data/emoji2vec_test.xlsx')[['content', 'label']]
    # e2v_path = 'Data/emoji2vec_data/emoji2vec.bin'
    # w2v_path = 'Data/emoji2vec_data/GoogleNews-vectors-negative300.bin.gz'
    # w2v = gsm.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    # e2v = gsm.models.KeyedVectors.load_word2vec_format(e2v_path, binary=True)

    # Process and split into three sets
    data['cleaned_content'] = data.content.apply(utils.preprocess_apply)
    test['cleaned_content'] = test.content.apply(utils.preprocess_apply)
    X, y = data['cleaned_content'].values, pd.get_dummies(data['label']).values.astype('float')
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SEED, test_size=0.2)
    X_test, y_test = test['cleaned_content'].values, pd.get_dummies(test['label']).values.astype('float')
    print(f'shape of train data is {X_train.shape}')
    print(f'shape of test data is {X_test.shape}')

    # Load the BERT model

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    bert = AutoModel.from_pretrained('roberta-base')

    # Create data loader
    # TweetTknzr = tk.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    MAX_LEN = 256
    BATCH_SIZE = 64

    train_data_loader = create_data_loader(X_train, y_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(X_val, y_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(X_test, y_test, tokenizer,  MAX_LEN, BATCH_SIZE)
    # Testing code for data loader

    # dataiter = iter(train_data_loader)
    # sample_inputs, sample_targets = dataiter.next()
    # print("Sample batch shape:", sample_inputs.shape, sample_targets.shape)

    # Set up the training hyperparams and the model
    hidden_dim = 512
    bidirectional = True
    EPOCHS = 15
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("You are training on", device)

    decoder = BERT_BiLSTM_FFF_Cls(bert, hidden_dim, bidirectional)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    # print(decoder)

    # Training model
    # history = defaultdict(list)
    best_accuracy = 0
    decoder.to(device)

    print('=======sanity test=======')
    val_acc, val_loss = eval_model(
        decoder,
        val_data_loader,
        loss_fn,
        device,
        len(X_val)
    )
    print(f'Val loss {val_loss} accuracy {val_acc}')
    print()
    print('========================')

    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            decoder,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            len(X_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            decoder,
            val_data_loader,
            loss_fn,
            device,
            len(X_val)
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        # # For visualization
        # history['train_acc'].append(train_acc)
        # history['train_loss'].append(train_loss)
        # history['val_acc'].append(val_acc)
        # history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(decoder.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    decoder.load_state_dict(torch.load('best_model_state.bin'))
    test_acc, test_loss = eval_model(decoder, test_data_loader,loss_fn,
            device,
            len(X_test))
    print(f"Best model's accuracy on test set:{test_acc}, loss {test_loss}")