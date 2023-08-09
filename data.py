from torch.utils.data import Dataset
import nltk
from utils import get_vocab
from PIL import Image
import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence

class ImageCaptionDataset(Dataset):
    def __init__(self,
                 transform = None,
                 df = None,
                 img_path = None,
                 load_vocab = False,
                 path_vocab = ""):
        
        super(Dataset, self).__init__()
        self.size = df.shape[0]
        self.df = df
        self.transform = transform
        self.tokenizer = nltk.tokenize.word_tokenize

        if load_vocab:
            with open(path_vocab, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            self.vocab = get_vocab(df, self.tokenizer)
            self.__save_vocab()
            
        self.img_path = img_path

    def char_to_id(self, character):
        return self.vocab[character]
    
    def __save_vocab(self):
        with open('myvocab.pkl', 'wb') as f:
                pickle.dump(self.vocab, f)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, self.df.iloc[idx]['filename']))
        caption = self.df.iloc[idx]["captions"]

        if self.transform:
            img = self.transform(img)

        tokens = self.tokenizer(caption)
        labels = []
        labels.append(self.vocab['<bos>'])
        labels.extend([self.vocab[token] for token in tokens])
        labels.append(self.vocab['<eos>'])
        labels = torch.Tensor(labels).long()

        return img, labels


def generate_batch(data_batch):
    
    PAD_IDX = 1
    
    if len(data_batch) == 1:
        return data_batch[0][0], data_batch[0][1]
    
    image_batch, text_batch = [], []

    for (img, text) in data_batch:
        image_batch.append(img)
        text_batch.append(text)

    image_batch = torch.stack(image_batch)
    text_batch = pad_sequence(text_batch, padding_value=PAD_IDX).T

    return image_batch, text_batch