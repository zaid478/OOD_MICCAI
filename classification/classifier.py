##  CLASSIFICATION MODEL FOR CNN.

# python classifier.py --train_path ../train_with_rotations/ --model_output models_classification/

import argparse
import torch
import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,models
from torch import nn
from torch.autograd import Variable
import numpy as np
import glob,os,time
import pandas as pd
from torch import optim
import pathlib
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from eval import fit_model
import torchvision

# Parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--train_path", required=True,
	help="path to Training data")
ap.add_argument("-m", "--model_output", required=True,
	help="path to directory where model will be saved")
ap.add_argument("-r", "--last_checkpoint", default=None,
	help="path to last saved checkpoint")

args = vars(ap.parse_args())

train_path = args['train_path']

# Network Architecture
class Densenet(nn.Module):
    def __init__(self, densenet, num_class, freeze_conv=False, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=1024,neurons_reducer_block=256):
        
        super(Densenet, self).__init__()

        self.features = nn.Sequential(*list(densenet.children())[:-1])
        
        self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv, 256),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        
        self.classifier = nn.Linear(256 , num_class)
    
    def forward(self,x):
        feat_ = self.features(x)
        feat_ = F.relu(feat_, inplace=True)
        feat_ = F.adaptive_avg_pool2d(feat_, (1, 1)).view(x.size(0), -1)
        feat_reducer = self.reducer_block(feat_)
        out_ = self.classifier(feat_reducer)
        return out_

# Pre processing transforms
transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# train contains 4 folders (one for each class)
dataset_malaria = datasets.ImageFolder(root = train_path,transform=transforms)

print (dataset_malaria.class_to_idx) # ("gametocyte":0,"ring":1,"schizont":2,"trophozoite":3)

# training and validation splits
dataset_size = len(dataset_malaria)
indices = list(range(dataset_size))
split = int(np.floor(0.25 * dataset_size))
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


# training and validation loaders
train_loader = torch.utils.data.DataLoader(dataset_malaria, batch_size=64, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset_malaria, batch_size=64,
                                                sampler=valid_sampler)

# Pre-trained model for finetuning
densenet = models.densenet121(pretrained=True)
model = Densenet(densenet, 4, neurons_reducer_block=256, freeze_conv=False,
                  comb_method=None, comb_config=None)

# weights for class-balanced cross entropy loss - More weight implies lesser number of samples
weights = [13755/875,13755/2926,13755/1064,13755/8890]
print(weights)
loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(weights).cuda())


model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-6,
patience=10)

epochs = 150
best_metric = "loss"
model_name = "Densenet"
model_path = args['last_checkpoint']
save_folder_ = args['model_output'] + model_name + "_fold_" + "_" + str(datetime.datetime.now()).replace(' ', '')

# Model fitting
fit_model(model, train_loader, validation_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs,
            epochs_early_stop=15, save_folder=save_folder_, initial_model=model_path,
            device=None, schedule_lr=scheduler_lr, config_bot=None, model_name="CNN", resume_train=True,
            history_plot=True, val_metrics=["auc"], best_metric=best_metric)
