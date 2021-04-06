# OOD detection training code - MICCAI paper

# For running OOD with pre-trained classification model on malaria dataset
## python malaria_ood.py --train_path train_with_rotations/ --model_output models/v1 --classifier_path
##  /home/iml/Desktop/Zaid/malaria_as_id/Code_clean_OOD/classification/models_classification/Densenet_fold__2021-04-0612:39:46.306055/best-checkpoint/best-checkpoint.pth

# For resuming OOD from previous checkpoint
## python malaria_ood.py --train_path train_with_rotations/ --model_output models/v1 --last_checkpoint models/v1/1.pth 


import numpy as np
import argparse
import torchvision
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,models
from torch import nn
from torch.autograd import Variable
import numpy as np
import glob,os,time,pathlib
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses

from densenet_121_base import Net



# Parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--train_path", required=True,
	help="path to Training data")
ap.add_argument("-m", "--model_output", required=True,
	help="path to directory where model will be saved")
ap.add_argument("-c", "--classifier_path", default=None,
	help="path to pre-trained classifier")
ap.add_argument("-r", "--last_checkpoint", default=None,
	help="path to last saved checkpoint")

args = vars(ap.parse_args())

if args['classifier_path'] is None and args['last_checkpoint'] is None:
   ap.error("at least one of --classifier_path and --last_checkpoint required")


if not os.path.exists(args['model_output']):
        print ('Model output folder does not exist. I am creating it!')
        os.mkdir(args['model_output'])

train_path = args['train_path']

# Pre processing transforms
transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# train contains 5 folders (four for dataset classes and one for OOD)
dataset_malaria = datasets.ImageFolder(root = train_path,transform=transforms)
train_loader = torch.utils.data.DataLoader(dataset_malaria, batch_size=64,shuffle=True)

# Training function
def train(model,loader_train,criterion,criterion_ce,optimizer,num_epochs=500,loader_test=None):

  model.train()
  losses_ = []
  best_loss = 100
  for epoch in range(1,num_epochs):
    st = time.time()
    # initializing variables for accumulating losses over iterations
    loss_epoch,loss_epoch_tuplet,loss_epoch_classification,loss_epoch_id_tuplet,loss_epoch_ood_tuplet = 0,0,0,0,0

    print ("Epoch ",epoch)
    for iter_,(anchor,label) in enumerate(loader_train):
    
      ## This assumes that OOD folder has class index 0 in the training folder. Do change it accordingly if index is different.
      batch_size = anchor.shape[0]
      
      ## Retreiving OOD and ID indexes 
      ood_indices = torch.where(label==0)[0]
      num_ood_labels = len(ood_indices)
      id_indices = torch.where(label!=0)[0]
      
      ## creating a new label list where 0 represents OOD samples and 1 represents ID samples
      new_label = label.detach().clone()
      new_label[id_indices] = 1
      new_label[ood_indices] = 0


      anchor = Variable(anchor).cuda()
      label = Variable(label).cuda()


      ## First output is the 256-dimensional vector and last one is the softmax
      output_anchor,last_anchor = model(anchor)
      output_anchor = output_anchor.double()
      last_anchor = last_anchor.double()

      ## extracting ID softmax vectors and corresponding class labels for classification loss
      last_id = last_anchor[id_indices]
      label_id = label[id_indices]
      
      ## extracting ID feature vectors (256-dim) and corresponding class labels for tuplet loss
      feat_id = output_anchor[id_indices]

      ## OOD tuplet loss where OOD is labelled as 0 and ID is labelled as 1. It will further OOD samples from ID samples.
      loss_ood_tuplet = criterion(output_anchor,new_label)

      ## ID tuplet loss which will cluster classwise ID samples
      loss_id_tuplet = criterion(feat_id,label_id)
      
      ## classification loss for all.
      loss_classification = criterion_ce(last_anchor,label)

      ## Total loss
      loss_total = loss_id_tuplet + loss_ood_tuplet + loss_classification 

        
      # ===================backward====================
      optimizer.zero_grad()
      loss_total.backward()
      optimizer.step()

      loss_epoch += loss_total.item()
      loss_epoch_id_tuplet += loss_id_tuplet.item()
      loss_epoch_classification += loss_classification.item()
      loss_epoch_ood_tuplet += loss_ood_tuplet.item()


        # ===================log========================
    avg_loss = loss_epoch/len(loader_train)
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch , num_epochs, avg_loss))
    losses_.append(loss_epoch/len(loader_train))
    print("ID Tuplet loss: ", loss_epoch_id_tuplet/len(loader_train))
    print("Classification loss: ", loss_epoch_classification/len(loader_train))
    print("OOD Tuplet loss: ", loss_epoch_ood_tuplet/len(loader_train))


    print ("epoch time: ",time.time()-st)
    if avg_loss < best_loss:
      best_loss =  avg_loss
      model_path = os.path.join(args['model_output'],str(epoch)+".pth")
      torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss_epoch/len(loader_train),
              'loss_tuplet': loss_epoch_tuplet/len(loader_train),
              'loss_tuplet_ood': loss_epoch_ood_tuplet/len(loader_train)
            }, 
              model_path,_use_new_zipfile_serialization=False)


    
    
  return losses_,model



# Losses
## Tuplet loss initialization
main_loss = losses.TupletMarginLoss()
var_loss = losses.IntraPairVarianceLoss()
criterion = losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
## Cross entropy loss
criterion_ce = nn.CrossEntropyLoss()



if args['last_checkpoint'] is None:
    # Initializing model object (we have 4 classes in malaria dataset and the classification network was trained on 4 classes)
    torch_model = Net(models.densenet121(pretrained=False),4)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    ## Loading model file
    ckpt = torch.load(args['classifier_path'])
    
    ## Some keys are not matched due to GPU issues, so we make sure to get rid of it. A bit of manual process.
    newstate = OrderedDict()
    for k,v in ckpt['model_state_dict'].items():
        t = True
        if k == "classifier.weight":
            n = "classifier.0.weight"
            newstate[n] = v
            t = False
        if k == "classifier.bias":
            n = "classifier.0.bias"
            newstate[n] = v
            t = False

        if t:
            newstate[k] = v

    torch_model.load_state_dict(newstate)

    ## Since we have OOD samples too, we will have 5 neurons in the output layer instead of four. Pytorch does not allow otherwise.
    torch_model.classifier =  nn.Sequential(
                    nn.Linear(256 , 5)
                )
    torch_model.cuda()

else:
    # Initializing model object (we have 4 classes in malaria dataset and 1 more for OOD)
    torch_model = Net(models.densenet121(pretrained=False),5)
    torch_model.cuda()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.0001, weight_decay=1e-5)

    ckpt = torch.load(args['last_checkpoint'])
    torch_model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])



print(torch_model)


# Calling the training function
loss,model_n = train(torch_model,train_loader,criterion,criterion_ce,optimizer,num_epochs=200)