# Feature extraction from trained OOD network

# python feature_extractor.py -d data_all/ -f features/v1 -m models/v1/1.pth
'''
data_all/ should have folders for every class. A dummy structure would be as follows:
  - Training set class 1 
  - Training set class 2 
  - Training set class 3
  - Training set class 4
  - test_id (In-distribution test set. No sub folders should be there. ID samples are considered one class here)
  - bbox
  - coin_fusion
  ...

All classes should be there including all training classes,all OOD classes/dataset and In-distribution datasets.

The code will dump all features of the "data_all" folders in the "features/v1" folder with the same names 

'''

import torch
import glob,os,pathlib,argparse
from torchvision import models,transforms
from PIL import Image
import numpy as np
import torch.nn.functional as nnF
from torch import nn
from collections import OrderedDict

# Parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_path", required=True,
	help="path to a folder containing Training data, Testing In-distribution data and Testing OOD data")
ap.add_argument("-f", "--feature_dir", required=True,
	help="path to directory where features will be saved")
ap.add_argument("-m", "--model_path", required=True,
	help="path to OOD detector")

args = vars(ap.parse_args())

# Network architecture
class Net (nn.Module):
    def __init__(self, densenet, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256,
                 feat_reducer=None, classifier=None):
        
        super(Net, self).__init__()
        
        self.features = nn.Sequential(*list(densenet.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        if feat_reducer is None:
            self.reducer_block = nn.Sequential(
                nn.Linear(1024, neurons_class),
                nn.BatchNorm1d(neurons_class),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.reducer_block = feat_reducer

        if classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(neurons_class + n_extra_info, num_class)
            )
        else:
            self.classifier = classifier

    def forward(self, img, extra_info=None):

        xf = self.features(img)
        x = nnF.relu(xf, inplace=True)
        x = nnF.adaptive_avg_pool2d(x, (1, 1)).view(xf.size(0), -1)

        x = self.reducer_block(x)

        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x

        x_out = self.classifier(agg)

        return x,x_out




# Initializing and loading model
torch_model = Net(models.densenet121(pretrained=False), 5)
torch_model_snap = torch.load(args['model_path'])
torch_model.load_state_dict(torch_model_snap['model_state_dict'])
torch_model.eval()

## Function to extract and save features
def extract_features_from_reducer(model,cls_name):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tranform_img = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])

    path_to_data = pathlib.Path(os.path.join(args['data_path'],cls_name))

    path_to_features = os.path.join(args['feature_dir'],cls_name)
    if not os.path.exists(path_to_features):
        os.mkdir(os.path.join(args['feature_dir'],cls_name))
    
    ## Looping over all images in the directory
    for img_path in path_to_data.iterdir():

        if ("rotated" in str(img_path)):
            continue
        image = Image.open(str(img_path)).convert("RGB")
        image = tranform_img(image)
        image = image.unsqueeze(0)
        feature,_ = model(image)

        feature = feature.detach().numpy()
        img_name = img_path.stem
        name_feature_file = path_to_features + "/" + img_name
        np.save(name_feature_file,feature)
    
    print ("DONE!")


## Looping over every folder in the data directory
for cls_ in os.listdir(args['data_path']):
    print (cls_)
    extract_features_from_reducer(torch_model,cls_)