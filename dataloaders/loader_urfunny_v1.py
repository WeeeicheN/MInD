import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

from create_dataset import Create_URFUNNY
import numpy as np
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

bert_tokenizer = AutoTokenizer.from_pretrained('/home/dwc/pretrained/bert-base-uncased')

UNK = 0
PAD = 1

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class URFUNNY(Dataset):
    def __init__(self, configs, mode='train'):
        DATA_PATH = configs.data_root

        if not os.path.isfile(DATA_PATH + '/train.pkl'):
            Create_URFUNNY(configs)

        if mode == "train":
            self.data = load_pickle(DATA_PATH + '/train.pkl')
        elif mode == "valid":
            self.data = load_pickle(DATA_PATH + '/dev.pkl')
        elif mode == "test":
            self.data = load_pickle(DATA_PATH + '/test.pkl')
        else:
            print("Mode is not set properly (train/valid/test)")
            exit()

        self.len = len(self.data)

        #configs.visual_size = self.data[0][0][1].shape[1]
        #configs.acoustic_size = self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class URFUNNY_Loader(object):
    def __init__(self, configs, train_drop_last=False, valid_drop_last=False, test_drop_last=False):
        super().__init__()
        self.configs = configs
        self.train_drop_last = train_drop_last
        self.valid_drop_last = valid_drop_last
        self.test_drop_last = test_drop_last
        self.train_dataset = URFUNNY(configs, mode='train')
        self.valid_dataset = URFUNNY(configs, mode='valid')
        self.test_dataset = URFUNNY(configs, mode='test')

    def seed_worker(self, worker_id):
        """
        Ensure ++ reproducible, usage: Dataloader(..., worker_init_fn=self.seed_worker, ...)
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)

        # Create bert indices using tokenizer
        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                            text, 
                            max_length=SENT_LEN+2, 
                            add_special_tokens=True, 
                            padding='max_length', 
                            truncation=True,
                            )
            bert_details.append(encoded_bert_sent)


        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        return visual, acoustic, labels, bert_sentences, bert_sentence_types, bert_sentence_att_mask
    
    @property
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.configs.batch_size,
                          shuffle=True,
                          num_workers=16,
                          pin_memory=True,
                          collate_fn=self.collate_fn,
                          drop_last=self.train_drop_last)
    
    @property
    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.configs.batch_size,
                          shuffle=False,
                          num_workers=16,
                          pin_memory=True,
                          collate_fn=self.collate_fn,
                          drop_last=self.valid_drop_last)

    @property
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.configs.batch_size,
                          shuffle=False,
                          num_workers=16,
                          pin_memory=True,
                          collate_fn=self.collate_fn,
                          drop_last=self.test_drop_last)