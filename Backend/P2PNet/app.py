# Copyright 2021 Tencent

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import numpy as np
import torch
import warnings
import random
import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.p2pnet import P2PNet
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


def predict(img):
    if img is None:
        return "No image selected", plt.figure()
    """the main process of inference"""
    test_loader = loading_data(img)
    #model = SASNet()
    model = P2PNet().cpu()
    model_path = "./weight/SHTechA.pth"
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


label, fig = predict(Image.open("IMG_1.jpg"))
print(label)
plt.show()