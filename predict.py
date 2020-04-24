import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
from networkUtil import prdictCli

import json

args = prdictCli()

def loading_model (file_path):
    checkpoint = torch.load (file_path) 
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else:
        model = models.vgg13 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False 

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        
    '''
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor


def predict(image_path, model, topkl, cuda):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available() and cuda:
       model.to("cuda")
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topkl)


def main():
    print(args.category_names)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model_path = args.save_dir + '/checkpoint.pth'    
    model = loading_model (model_path)
    print("loading model complete")
    probs, classes = predict (args.image_path, model, args.top_k, args.gpu)
    print("predict complete")
    props_ar = np.array(probs[0])
    class_names = [cat_to_name[str(index + 1)] for index in np.array(classes[0])]
    for l in range (len(class_names)):
         print("Number: {}. ".format(l+1),
                "Class name: {}.. ".format(class_names[l]),
                "Probability: {:.3f}..% ".format(props_ar[l]*100),
                )
            
if __name__== "__main__":
    main()
    
 #python predict.py flowers/test/1/image_06743.jpg