import cv2
import numpy as np
from Calibration import CalibrationYOLO
from P2PNet import PersistentP2P
from typing import Tuple, List
# from numba import cuda, njit # <= 0.56
import torch
import gc
import os
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class MagicFrameProcessor:
    def __init__(self) -> None:
        self.is_calibrating = False
        self.mode = 0  # p2p mode
        self.__calibration = None  # calibration obj containing YOLOv8
        self.__magic = None  # scale func
        self.__magic_weight = 0  # scale func weight for re-calibration

        self.__p2p = None

    def process(self, frame: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        if not calibrated -> calibrate
         :store magic_func :return annotated image
        else -> p2p -> heatmap
         :store sampling :return count, heatmap
        """
        # self.__magic = lambda x: 0.00445804253434011*x+0.1
        width, height = frame.shape[1], frame.shape[0]
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        frame = cv2.resize(frame, dsize=(new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        if not self.__magic:  # change todo re-calibration
            return self.__calibrate(frame=frame)

        count, head_coords = self.__find_heads(frame=frame)
        # heatmap = self.__create_heatmap(frame, head_coords, overlay=True)
        heatmap = self.__show_heads(frame, head_coords)
        return count, heatmap

    def __calibrate(self, frame: np.ndarray, sample_points=5):
        self.is_calibrating = True
        if not self.__calibration:
            args = dict({'test': True, 'average_height': 173})
            frame_wh = frame.shape[1], frame.shape[0]  # float width , height
            self.__calibration = CalibrationYOLO(args, *frame_wh)

        res = self.__calibration.extract_entities(frame=frame)
        annotated_frame = res[0].plot()

        if self.__calibration.size >= sample_points:
            self.is_calibrating = False
            self.__magic_weight += 1
            a, b = self.__calibration.create_lerp_function()
            print('##############################', a, b)
            self.__magic = lambda x: a*x+b
            self.mode = self.__calibration.mode
            self.__calibration = None
            gc.collect()
            torch.cuda.empty_cache()

            # do the magic func merge dance here probably..
        return -1, annotated_frame

    def __find_heads(self, frame: np.ndarray):
        if not self.__p2p:
            self.__p2p = PersistentP2P()
        return self.__p2p.process(frame=frame)

    def __show_heads(self, frame: np.ndarray, points: List[List[float]]):
        for p in points:
            img_to_draw = cv2.circle(
                frame, (int(p[0]), int(p[1])), int(2), (0, 0, 255), -1)
        return img_to_draw

    def __create_heatmap(self, frame: np.ndarray, points: List[List[float]], overlay: bool = False) -> np.ndarray:
        # draw the predictions
        size = 10
        img_to_draw = np.zeros(frame.shape, np.uint8)
        magic = self.__magic
        for p in points:
            scaled = magic(p[1]) * size
            img_to_draw = cv2.circle(
                img_to_draw, (int(p[0]), int(p[1])), int(scaled), (255, 255, 255), -1)

        blur = cv2.GaussianBlur(img_to_draw, (13, 13), 11)
        heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

        if overlay:
            return cv2.addWeighted(heatmap, 0.5, frame, 0.5, 0)

        return heatmap


if __name__ == '__main__':
    cap = cv2.VideoCapture('istock-962060884_preview.mp4')
    magic = MagicFrameProcessor()
    while True:
        success, frame = cap.read()
        if success:
            count, img = magic.process(frame=frame)

            cv2.imshow("YOLOv8 Inference", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    cap.release()
    cv2.destroyAllWindows()
