import numpy as np
import random
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

bert_tokenizer = AutoTokenizer.from_pretrained('/home/dwc/pretrained/bert-base-uncased')

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class InputFeatures(object):
    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class MOSI:
    def __init__(self, configs, mode='train'):
        DATA_PATH = configs.data_root

        self.data = load_pickle(DATA_PATH + '/mosi.pkl')
        if mode == "train":
            self.data = self.data["train"]
        elif mode == "valid":
            self.data = self.data["dev"]
        elif mode == "test":
            self.data = self.data["test"]
        else:
            print("Mode is not set properly (train/valid/test)")
            exit()

        self.dataset = self.get_dataset(self.data, configs)

    def get_dataset(self, data, configs):

        features = self.convert_to_features(data, configs.max_seq_length, bert_tokenizer, configs)
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long)
        all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
        all_acoustic = torch.tensor(
            [f.acoustic for f in features], dtype=torch.float)
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)

        all_label_ids = all_label_ids.squeeze(1)
        #print(all_label_ids.shape)

        sentences = torch.randn_like(all_label_ids)
        lengths = torch.randn_like(all_label_ids)

        dataset = TensorDataset(
            all_visual,
            all_acoustic,
            all_label_ids,
            all_input_ids,
            all_segment_ids,
            all_input_mask,
            )
        
        return dataset

    def convert_to_features(self, examples, max_seq_length, tokenizer, configs):
        features = []

        for (ex_index, example) in enumerate(examples):

            (words, visual, acoustic), label_id, segment = example
            # print(words)
            tokens, inversions = [], []
            for idx, word in enumerate(words):
                tokenized = tokenizer.tokenize(word)
                # print(tokenized)
                tokens.extend(tokenized)
                inversions.extend([idx] * len(tokenized))

            # Check inversion
            assert len(tokens) == len(inversions)

            aligned_visual = []
            aligned_audio = []

            for inv_idx in inversions:
                aligned_visual.append(visual[inv_idx, :])
                aligned_audio.append(acoustic[inv_idx, :])

            visual = np.array(aligned_visual)
            acoustic = np.array(aligned_audio)

            # Truncate input if necessary
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[: max_seq_length - 2]
                acoustic = acoustic[: max_seq_length - 2]
                visual = visual[: max_seq_length - 2]

            input_ids, visual, acoustic, input_mask, segment_ids = self.prepare_input(
                tokens, visual, acoustic, tokenizer, configs
            )

            # Check input length
            assert len(input_ids) == configs.max_seq_length
            assert len(input_mask) == configs.max_seq_length
            assert len(segment_ids) == configs.max_seq_length
            assert acoustic.shape[0] == configs.max_seq_length
            assert visual.shape[0] == configs.max_seq_length

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    visual=visual,
                    acoustic=acoustic,
                    label_id=label_id,
                )
            )
        return features
    
    def prepare_input(self, tokens, visual, acoustic, tokenizer, configs):
        ACOUSTIC_DIM = 74
        VISUAL_DIM = 47

        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        tokens = [CLS] + tokens + [SEP]

        # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
        acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
        acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
        visual_zero = np.zeros((1, VISUAL_DIM))
        visual = np.concatenate((visual_zero, visual, visual_zero))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        pad_length = configs.max_seq_length - len(input_ids)

        acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
        acoustic = np.concatenate((acoustic, acoustic_padding))

        visual_padding = np.zeros((pad_length, VISUAL_DIM))
        visual = np.concatenate((visual, visual_padding))

        padding = [0] * pad_length

        # Pad inputs
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return input_ids, visual, acoustic, input_mask, segment_ids

class MOSI_Loader(object):
    def __init__(self, configs, train_drop_last=True, valid_drop_last=False, test_drop_last=False):
        super().__init__()
        self.configs = configs
        self.train_drop_last = train_drop_last
        self.valid_drop_last = valid_drop_last
        self.test_drop_last = test_drop_last
        self.train_dataset = MOSI(configs, mode='train').dataset
        self.valid_dataset = MOSI(configs, mode='valid').dataset
        self.test_dataset = MOSI(configs, mode='test').dataset
    
    def seed_worker(self, worker_id):
        """
        Ensure ++ reproducible, usage: Dataloader(..., worker_init_fn=self.seed_worker, ...)
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    @property
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.configs.batch_size,
                          shuffle=True,
                          num_workers=16,
                          pin_memory=True,
                          drop_last=self.train_drop_last)
    
    @property
    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.configs.batch_size,
                          shuffle=False,
                          num_workers=16,
                          pin_memory=True,
                          drop_last=self.valid_drop_last)

    @property
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.configs.batch_size,
                          shuffle=False,
                          num_workers=16,
                          pin_memory=True,
                          drop_last=self.test_drop_last)