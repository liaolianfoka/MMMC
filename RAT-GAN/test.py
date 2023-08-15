import pickle
import torch
"""
with open("../data/test/captions.pickle", "rb") as f:
    data = pickle.load(f)
    s1 = []
    s2 = []
    for line in data[0]:
        line = line.tolist()
        if line not in s1:
            s1.append(line)
    for line in data[1]:
        line = line.tolist()
        if line not in s2:
            s2.append(line)
    cnt = 0
    for line in s1:
        if line in s2:
            cnt += 1
    for line in s1:
        print(line[:3])
    print("-----------------")
    for line in s2:
        print(line[:3])
    print(len(data[0]))
"""
torch.manual_seed(100)
print(torch.randn(2))
print(torch.randn(2))