import csv, pickle, re
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def load_dataset(path, args):
    contents = []

    with open("../data/Eimg2emb.pkl", 'rb') as f: 
        Eimg2emb = pickle.load(f)

    with open("../data/Eimg2vit.pkl", "rb") as f:
        Eimg2vit = pickle.load(f)

    with open("../data/text2emb.pkl", 'rb') as f:
        text2emb = pickle.load(f)
    
    with open("../data/Fimg2emb.pkl", 'rb') as f: 
        Fimg2emb = pickle.load(f)

    with open("../data/Fimg2vit.pkl", "rb") as f:
        Fimg2vit = pickle.load(f)
        Fimg2vit[0] = np.zeros(512)

    with open("../data/source2emb.pkl", 'rb') as f:
        source2emb = pickle.load(f)

    with open("../data/target2emb.pkl", 'rb') as f:
        target2emb = pickle.load(f)

    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row, line in enumerate(reader):
            if row==0:
                continue
            id = int(line[0].split("(")[1].split(")")[0])
            S = int(line[7]) if line[7] not in ["","None", "none"] else 0
            T = int(line[8]) if line[8] not in ["","None", "none"] else 0
            label = int(line[args.label].split("(")[0])
            if args.label==4:
                label += 1
            Eimgemb = Eimg2emb[id]
            Eimgvit = Eimg2vit[id]
            Simgemb = Fimg2emb[S]
            Simgvit = Fimg2vit[S]
            Timgemb = Fimg2emb[T]
            Timgvit = Fimg2vit[T]
            textfeature = text2emb[id]
            sourcefeature = source2emb[id]
            targetfeature = target2emb[id]
            contents.append([Eimgemb, Eimgvit, Simgemb, Simgvit, Timgemb, Timgvit, textfeature, sourcefeature, targetfeature, label - 1])
    return contents

def build_dataset(args):
    train_path = "../data/train.csv"
    val_path = "../data/val.csv"
    test_path = "../data/test.csv"

    train = load_dataset(train_path, args)
    val = load_dataset(val_path, args)
    test = load_dataset(test_path, args)

    return train, val, test

# 构建迭代器
class DatasetIterater(object):  # 自定义数据集迭代器
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches  # 构建好的数据集
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:  # 不能整除
            self.residue = True #True表示不能整除
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        #Eimgemb, Eimgvit, Simgemb, Simgvit, Timgemb, Timgvit, textfeature, sourcefeature, targetfeature
        Eimgemb = torch.FloatTensor([_[0].tolist() for _ in datas]).to(self.device)  # 32*2048
        Eimgvit = torch.FloatTensor([_[1].tolist() for _ in datas]).to(self.device)
        Simgemb = torch.FloatTensor([_[2].tolist() for _ in datas]).to(self.device)
        Simgvit = torch.FloatTensor([_[3].tolist() for _ in datas]).to(self.device)
        Timgemb = torch.FloatTensor([_[4].tolist() for _ in datas]).to(self.device)
        Timgvit = torch.FloatTensor([_[5].tolist() for _ in datas]).to(self.device)
        textfeature = torch.FloatTensor([_[6].tolist() for _ in datas]).to(self.device)
        sourcefeature = torch.FloatTensor([_[7].tolist() for _ in datas]).to(self.device)
        targetfeature = torch.FloatTensor([_[8].tolist() for _ in datas]).to(self.device)

        y = torch.LongTensor([_[9] for _ in datas]).to(self.device)
        return (Eimgemb, Eimgvit, Simgemb, Simgvit, Timgemb, Timgvit, textfeature, sourcefeature, targetfeature), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:  #当数据集大小不整除 batch_size时，构建最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)  # 把最后一个batch转换为tensor 并 to(device)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:  # 构建每一个batch
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)  # 把当前batch转换为tensor 并 to(device)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue: #不能整除
            return self.n_batches + 1 #batch数+1
        else:
            return self.n_batches

def build_iterator(dataset, batch_size, device): #构建数据集迭代器
    iter = DatasetIterater(dataset, batch_size, device)
    return iter