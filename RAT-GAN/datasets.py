#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


# In[ ]:


def get_imgs(img_path, imsize, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    ret = []
    ret.append(normalize(img))
    return ret


# In[ ]:


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None, bert_path=None, device="cpu"):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),#转为tensor，并归一化至[0-1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#按通道进行标准化，先减0.5，再除以0.5
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.bert_path = bert_path
        self.device = device
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):#默认为1
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions = self.load_text_data(data_dir, split)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
    
    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            if cfg.SINGLE:
                 cap_path = '%s/text/%s.txt' % ("/".join(data_dir.split("/")[:-1]), filenames[i])
            else:
                cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf-8').decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ").lower()
                    all_captions.append(cap)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions
    
    def get_tokens_bert(self, train_captions, test_captions):

        tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        bert = BertModel.from_pretrained(self.bert_path).to(self.device)
       
        for p in bert.parameters():
            p.requires_grad = False
        bert.eval()
        
        train_captions_new = []
        for t in train_captions:
            tokenized_text = tokenizer.tokenize("[CLS] " + t + " [SEP]")
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            out = bert(tokens_tensor)[1][0]
            train_captions_new.append(out.cpu())
        test_captions_new = []
        for t in test_captions:
            tokenized_text = tokenizer.tokenize("[CLS] " + t + " [SEP]")
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            out = bert(tokens_tensor)[1][0]
            test_captions_new.append(out.cpu())

        return [train_captions_new, test_captions_new]
    
    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)
            train_captions, test_captions = self.get_tokens_bert(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = [t[:cfg.TEXT.EMBEDDING_DIM] for t in x[0]], [t[:cfg.TEXT.EMBEDDING_DIM] for t in x[1]]
                del x
                print('Load from: ', filepath)
        if split == 'train':
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions
    
    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        return self.captions[sent_ix].cuda()

    def __getitem__(self, index):

        key = self.filenames[index]
        cls_id = self.class_id[index]
        data_dir = self.data_dir

        if cfg.SINGLE:
            img_name = '../data/img_data/%s.jpg' % (key)
        else:
            img_name = '../data/img_data/%s.jpg' % (key)
        imgs = get_imgs(img_name, self.imsize, self.transform, normalize=self.norm)
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps = self.get_caption(new_sent_ix)
        return imgs, caps, cls_id, key


    def __len__(self):
        return len(self.filenames)

