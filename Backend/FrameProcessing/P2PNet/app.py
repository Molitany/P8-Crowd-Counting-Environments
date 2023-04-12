import argparse

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from .engine import *
from .models import build_model
import os
get_path = lambda *x : os.path.join(os.path.dirname(__file__),*x)
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='./logs/',
                        help='path where to save')
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def run(args, img):
    device_name = 'cpu'
    device = torch.device(device_name) #TODO: add gpu support here
    # get the P2PNet
    model = build_model(args)
    # move to cpu/gpu
    model.to(device)
    # load trained model
    if get_path('weights/SHTechA.pth') is not None:
        checkpoint = torch.load(get_path('weights/SHTechA.pth'), map_location=device_name) 
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load the images
    img_raw = Image.fromarray(np.uint8(img)).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    return predict_cnt, img_to_draw
     #cv2.imwrite(os.path.join("./logs/", 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

def main(img):
    """
    Given an image path it runs P2PNet using the vgg model in the P2PNet folder and returns count and img as cv2 array
    """
    # arguments are ignored and default are used instead
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    return run(args, img)

