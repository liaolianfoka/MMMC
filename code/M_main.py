import time, os
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
from datetime import timedelta
import argparse
from M_loaddata import build_dataset
from M_loaddata import build_iterator
from M_model import CEC_TF
from M_train import train
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest="gpu", type=int, default=0)
    parser.add_argument("--label", dest="label", type=int)
    parser.add_argument("--save_file", dest="save_file", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    train_data, val_data, test_data = build_dataset(args)
    train_iter = build_iterator(train_data, batch_size=40, device=device)
    val_iter = build_iterator(val_data, batch_size=40, device=device)
    test_iter = build_iterator(test_data, batch_size=40, device=device)
    model = CEC_TF(args).to(device)
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    train(model, train_iter, val_iter, test_iter, device, args, loss_func)
  