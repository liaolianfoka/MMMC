import os
import pickle
from PIL import Image
import transformers
import torchvision.transforms as transforms
import numpy as np
from transformers import DistilBertModel, DistilBertTokenizer
import torch
from transformers import AutoProcessor, AutoModel
import csv
###creat Eimg2emb.pkl###
data = {}
transforms1 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
for file in os.listdir("../data/Eimages"):
    path = "../data/Eimages/" + file
    id = file.split(")")[0].split("(")[1]
    id = int(id)
    img = Image.open(path).convert("RGB")
    img1 = transforms1(img)
    data[id] = img1
with open("../data/Eimg2emb.pkl", "wb") as f:
    pickle.dump(data, f, -1)

###creat Fimg2emb.pkl###
data = {}
for file in os.listdir("../data/Fimages"):
    path = "../data/Fimages/" + file
    id = file.split("_")[0]
    while id[0] == "0":
        id = id[1:]
    id = int(id)
    img = Image.open(path).convert("RGB")
    img1 = transforms1(img)
    data[id] = img1
    
data[0] = np.zeros((3,224,224))
with open("../data/Fimg2emb.pkl", "wb") as f:
    pickle.dump(data, f, -1)

###creat text2emb.pkl###
model_class, tokenizer_class, pretrained_weights = (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
data = {}
torch.cuda.set_device(1)
for csv_file in ["train.csv", "val.csv", "test.csv"]:
        with open("data/" + csv_file) as f:
            reader = csv.reader(f)
            for row, line in enumerate(reader):
                if row == 0:
                    continue
                id = int(line[0].split("(")[1].split(")")[0])
                text = line[1]
                if text == "":
                    input_ids = tokenizer.encode("None", add_special_tokens=True)
                else:
                    input_ids = tokenizer.encode(text, add_special_tokens=True)
                emb = model(torch.tensor([input_ids]))[0][0,0,:]
                emb.squeeze().cpu().detach().numpy()
                data[id] = emb.squeeze().cpu().detach().numpy()
with open("data/text2emb", "wb") as f:
    pickle.dump(data, f, -1)

###creat source2emb###
data = {}
for csv_file in ["train.csv", "val.csv", "test.csv"]:
        with open("data/" + csv_file) as f:
            reader = csv.reader(f)
            for row, line in enumerate(reader):
                if row == 0:
                    continue
                id = int(line[0].split("(")[1].split(")")[0])
                text = line[5]
                if text == "":
                    input_ids = tokenizer.encode("None", add_special_tokens=True)
                else:
                    input_ids = tokenizer.encode(text, add_special_tokens=True)
                emb = model(torch.tensor([input_ids]))[0][0,0,:]
                emb.squeeze().cpu().detach().numpy()
                data[id] = emb.squeeze().cpu().detach().numpy()
with open("data/source2emb", "wb") as f:
    pickle.dump(data, f, -1)

###creat target2emb.pkl###
data = {}
for csv_file in ["train.csv", "val.csv", "test.csv"]:
        with open("data/" + csv_file) as f:
            reader = csv.reader(f)
            for row, line in enumerate(reader):
                if row == 0:
                    continue
                id = int(line[0].split("(")[1].split(")")[0])
                text = line[6]
                if text == "":
                    input_ids = tokenizer.encode("None", add_special_tokens=True)
                else:
                    input_ids = tokenizer.encode(text, add_special_tokens=True)
                emb = model(torch.tensor([input_ids]))[0][0,0,:]
                emb.squeeze().cpu().detach().numpy()
                data[id] = emb.squeeze().cpu().detach().numpy()
with open("data/target2emb.pkl", "wb") as f:
    pickle.dump(data, f, -1)

###creat Eimg2emb_clip.pkl###
processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
model = AutoModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
img2emb = {}

for file in os.listdir("../data/Eimages"):
    img_path =  "../data/Eimages/" + file
    id = file.split(")")[0].split("(")[1]
    id = int(id)
    print(img_path)
    img = Image.open(img_path)
    image_inputs = processor(images=img, return_tensors="pt")
    image_emb = model.get_image_features(**image_inputs)
    image_emb /= image_emb.norm(p=2, dim=-1, keepdim=True)
    image_emb = image_emb.detach().numpy()[0]
    img2emb[id] = image_emb

with open("../data/Eimg2emb_clip.pkl", "wb") as f:
    pickle.dump(img2emb, f, -1)

###creat Fimg2emb_clip.pkl###
processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
model = AutoModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
img2emb = {}

for file in os.listdir("../data/Fimages"):
    img_path =  "../data/Fimages/" + file
    id = file.split("_")[0]
    while id[0] == "0":
        id = id[1:]
    id = int(id)
    print(img_path)
    img = Image.open(img_path)
    image_inputs = processor(images=img, return_tensors="pt")
    image_emb = model.get_image_features(**image_inputs)
    image_emb /= image_emb.norm(p=2, dim=-1, keepdim=True)
    image_emb = image_emb.detach().numpy()[0]
    img2emb[id] = image_emb

with open("../data/Fimg2emb_clip.pkl", "wb") as f:
    pickle.dump(img2emb, f, -1)
