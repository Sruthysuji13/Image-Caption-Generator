import os
import re
import random
from collections import Counter
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.freq     = Counter()

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text):
        return re.findall(r'\b[a-z]+\b', text.lower())

    def build(self, captions, min_freq=config.MIN_WORD_FREQ):
        for caption in captions:
            self.freq.update(self.tokenize(caption))
        for word, count in self.freq.items():
            if count >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
        print(f'[Vocab] Built vocabulary: {len(self)} tokens')

    def encode(self, caption):
        tokens = self.tokenize(caption)
        return [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]

    def decode(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if word in ('<SOS>', '<PAD>'): continue
            if word == '<EOS>': break
            words.append(word)
        return ' '.join(words)


def load_captions(caption_file):
    captions = {}
    with open(caption_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',', 1)
            if len(parts) != 2: continue
            img_name, caption = parts
            img_name = img_name.split('#')[0].strip()
            captions.setdefault(img_name, []).append(caption.strip())
    print(f'[Data] Loaded captions for {len(captions)} images.')
    return captions


def split_data(captions):
    keys = list(captions.keys())
    random.seed(42)
    random.shuffle(keys)
    n       = len(keys)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)
    train_keys = keys[:n_train]
    val_keys   = keys[n_train:n_train + n_val]
    test_keys  = keys[n_train + n_val:]
    print(f'[Data] Split → train: {len(train_keys)} | val: {len(val_keys)} | test: {len(test_keys)}')
    return train_keys, val_keys, test_keys


class Flickr8kDataset(Dataset):
    def __init__(self, image_keys, captions, vocab, transform=None):
        self.vocab     = vocab
        self.transform = transform
        self.samples   = []
        for img_name in image_keys:
            img_path = os.path.join(config.IMAGE_DIR, img_name)
            if not os.path.exists(img_path): continue
            for caption in captions[img_name]:
                encoded = (
                    [vocab.word2idx['<SOS>']]
                    + vocab.encode(caption)
                    + [vocab.word2idx['<EOS>']]
                )
                self.samples.append((img_path, torch.tensor(encoded, dtype=torch.long)))
        print(f'[Dataset] {len(self.samples)} image-caption pairs loaded.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption


def collate_fn(batch):
    images, captions = zip(*batch)
    images   = torch.stack(images, dim=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions


def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(config.PIXEL_MEAN, config.PIXEL_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.PIXEL_MEAN, config.PIXEL_STD),
    ])


def get_loaders():
    captions = load_captions(config.CAPTION_FILE)
    vocab    = Vocabulary()
    vocab.build([cap for caps in captions.values() for cap in caps])
    train_keys, val_keys, test_keys = split_data(captions)
    train_set = Flickr8kDataset(train_keys, captions, vocab, get_transform(True))
    val_set   = Flickr8kDataset(val_keys,   captions, vocab, get_transform(False))
    test_set  = Flickr8kDataset(test_keys,  captions, vocab, get_transform(False))
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader, vocab