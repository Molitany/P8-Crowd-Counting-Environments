from .app import get_args_parser, build_model, get_path, standard_transforms
from typing import List, Tuple
from PIL import Image
import numpy as np
import argparse
import torch

class PersistentP2P:
    def __init__(self) -> None:
        """ self.model self.transform self.device"""
        parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
        args = parser.parse_args()
        
        use_gpu = torch.cuda.is_available()
        device_name = 'cuda' if use_gpu else 'cpu'
        self.device = torch.device(device_name)
        # get the P2PNet
        self.model = build_model(args)
        # move to cpu/gpu
        self.model.to(self.device)
        # load trained model
        if get_path('weights','SHTechA.pth') is not None:
            checkpoint = torch.load(get_path('weights','SHTechA.pth'), map_location=device_name) 
            self.model.load_state_dict(checkpoint['model'])
        # convert to eval mode
        self.model.eval()
        # create the pre-processing transform
        self.transform = standard_transforms.Compose([
            standard_transforms.ToTensor(), 
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def process(self, frame:np.ndarray) -> Tuple[int, List[List[float]]]:
        # load the images
        img_raw = Image.fromarray(np.uint8(frame)).convert('RGB')
        # round the size
        # pre-proccessing
        img = self.transform(img_raw)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(self.device)
        # run inference
        outputs = self.model(samples)
        outputs_scores = torch.nn.functional.softmax(
            outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        return predict_cnt, points

