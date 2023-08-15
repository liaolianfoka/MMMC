import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torch import nn
from PIL import Image
import transformers as ppb

class CEC_TF(nn.Module):
    def __init__(self, args):
        super(CEC_TF, self).__init__()
        self.swin_trans = models.swin_transformer.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        self.swin_trans = nn.Sequential(*list(self.swin_trans.children()))[:-1]
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(768, 512)
        self.fc4 = nn.Linear(768, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(768, 512)
        self.fc7 = nn.Linear(768, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(768, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(768, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(768, momentum=0.5)
        self.bn5 = nn.BatchNorm1d(512, momentum=0.5)
        self.bn6 = nn.BatchNorm1d(768, momentum=0.5)
        self.bn7 = nn.BatchNorm1d(512, momentum=0.5)
        self.bn8 = nn.BatchNorm1d(768, momentum=0.5)
        self.bn9 = nn.BatchNorm1d(768, momentum=0.5)
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tranformencoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=2)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=512, nhead=16)
        self.tranformencoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=2)
        self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tranformencoder3 = nn.TransformerEncoder(self.encoder_layer1, num_layers=2)
        self.encoder_layer4 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tranformencoder4 = nn.TransformerEncoder(self.encoder_layer1, num_layers=2)
        self.encoder_layer5 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tranformencoder5 = nn.TransformerEncoder(self.encoder_layer1, num_layers=2)
        self.relu = nn.ReLU()
        if args.label == 2:
            self.fc10 = nn.Linear(512, 7)
        elif args.label == 3:
            self.fc10 = nn.Linear(512, 5)
        elif args.label == 4:
            self.fc10 = nn.Linear(512, 4)

    def forward(self, Eimgemb, Eimgvit, Simgemb, Simgvit, Timgemb, Timgvit, textfeature, sourcefeature, targetfeature):
        Eimg_swim = self.swin_trans(Eimgemb)
        Eimg_swim = self.bn1(Eimg_swim)
        Eimg_vit = self.bn2(Eimgvit)
        textfeature = self.bn3(textfeature)

        Simg_swim = self.swin_trans(Simgemb)
        Simg_swim = self.bn4(Simg_swim)
        Simg_vit = self.bn5(Simgvit)
        sourcefeature = self.bn8(sourcefeature)

        Timg_swim = self.swin_trans(Timgemb)
        Timg_swim = self.bn6(Timg_swim)
        Timg_vit = self.bn7(Timgvit)
        targetfeature = self.bn9(targetfeature)

        Eimg_swim = F.dropout(self.fc1(Eimg_swim).unsqueeze(1), 0.3)
        Eimg_vit = F.dropout(self.fc2(Eimg_vit).unsqueeze(1), 0.3)
        textfeature = F.dropout(self.fc3(textfeature).unsqueeze(1), 0.3)
        Simg_swim = F.dropout(self.fc4(Simg_swim).unsqueeze(1), 0.3)
        Simg_vit = F.dropout(self.fc5(Simg_vit).unsqueeze(1), 0.3)
        sourcefeature = F.dropout(self.fc6(sourcefeature).unsqueeze(1), 0.3)
        Timg_swim = F.dropout(self.fc7(Timg_swim).unsqueeze(1), 0.3)
        Timg_vit = F.dropout(self.fc8(Timg_vit).unsqueeze(1), 0.3)
        targetfeature = F.dropout(self.fc9(targetfeature).unsqueeze(1), 0.3)

        E_fusion = torch.cat((Eimg_swim, Eimg_vit, textfeature), dim = 1)
        E_fusion = F.dropout(self.tranformencoder3(E_fusion),0.3)
        E_fusion = E_fusion.mean(1).unsqueeze(1)

        T_fusion = torch.cat((Timg_swim, Timg_vit, targetfeature), dim = 1)
        T_fusion = F.dropout(self.tranformencoder4(T_fusion),0.3)
        T_fusion = T_fusion.mean(1).unsqueeze(1)

        S_fusion = torch.cat((Simg_swim, Simg_vit, sourcefeature), dim = 1)
        S_fusion = F.dropout(self.tranformencoder5(S_fusion),0.3)
        S_fusion = S_fusion.mean(1).unsqueeze(1)

        fusion = torch.cat((E_fusion, S_fusion, T_fusion), dim = 1)
        fusion = F.dropout(self.tranformencoder1(fusion), 0.3)
        fusion = F.dropout(self.tranformencoder2(fusion), 0.3)
        fusion = fusion.mean(1)
        output = F.softmax(self.fc10(fusion))
        return output
