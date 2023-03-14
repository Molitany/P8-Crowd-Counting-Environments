# Based on SASNet from Tencent 2021 using the Apache License, Version 2.0

import os
import cv2
import numpy as np
import torch
import warnings
import random
import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import SASNet
from PIL import Image

warnings.filterwarnings('ignore')

# define the GPU id to be used
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class data(Dataset):
    def __init__(self, img, transform=None):
        self.image = img
        self.transform = transform

    def __len__(self):
        return 1000

    def __getitem__(self, x):
        # open image here as PIL / numpy
        image = self.image
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
      
        image = torch.Tensor(image)
        return image

def loading_data(img):
    # the augumentations
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    # dcreate  the dataset
    test_set = data(img=img, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    return test_loader


def predict(img, model_path):
    if img is None:
        return "No image selected", plt.figure()
    """the main process of inference"""
    test_loader = loading_data(img)
    #model = SASNet()
    model = SASNet().cpu()
    # load the trained model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print('successfully load model from', model_path)

    with torch.no_grad():
        model.eval()

        for vi, data in enumerate(test_loader, 0):
          img = data
          #img = img.cuda()
          img = img.cpu()
          pred_map = model(img)
          pred_map = pred_map.data.cpu().numpy()
          for i_img in range(pred_map.shape[0]):
            pred_cnt = np.sum(pred_map[i_img]) / 1000

            den_map = np.squeeze(pred_map[i_img])
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(den_map, aspect='auto')
            
            return int(np.round(pred_cnt, 0)), fig

def subtract_images(img1_path, img2_path):
    """
    Subtract two images from each other while being able to handle images of different sizes. Sparse to dense will create many light areas and the reverse will create darker images.\n
    Using cv2.subtract to take the difference instead of cv2.absdiff as we want to know the current image and not the reference image and current image.
    """
    # read images from path
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    # take height and width
    height1, width1 = img1.shape[0],img1.shape[1]
    height2, width2 = img2.shape[0],img2.shape[1]    
    # find minimum of height and widthe
    min_height = min(height1, height2)
    min_width = min(width1, width2)
    # take image pixels until minimum height and width
    img11 = img1[0:min_height, 0:min_width]
    img22 = img2[0:min_height, 0:min_width]
    # subtract the now equal sized images
    img_sub = cv2.subtract(img11, img22)
    # convert image to PIL format and convert to PIL image for SASNet
    img = cv2.cvtColor(img_sub, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

sub_img = subtract_images("Crosswalk/frame0.jpg", "Crosswalk/frame100.jpg")
sub_img.show()
labelA, fig = predict(sub_img, "SHHA.pth")
labelB, fig = predict(sub_img, "SHHB.pth")
print(f"SHHA: {labelA}, SHHB: {labelB}")
plt.show()