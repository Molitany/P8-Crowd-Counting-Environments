import cv2
import numpy as np
from Calibration import CalibrationYOLO
from P2PNet import PersistentP2P
from typing import Tuple, List

class MagicFrameProcessor:
    def __init__(self) -> None:
        self.is_calibrating = False
        self.mode = 0 # p2p mode
        self.__calibration = None # calibration obj containing YOLOv8
        self.__magic = None # scale func
        self.__magic_weight = 0 # scale func weight for re-calibration

        self.__p2p = None

    def process(self, frame:np.ndarray) -> Tuple(int, np.ndarray):
        """
        if not calibrated -> calibrate
         :store magic_func :return none
        else -> p2p -> heatmap
         :store sampling :return heatmap
        """
        if not self.__magic: # change todo re-calibration
            return self.__calibrate(frame=frame)
        
        count, head_coords = self.__find_heads(frame=frame)
        heatmap = self.__create_heatmap(head_coords)
        return count, heatmap

    def __calibrate(self, frame, sample_points=6):
        self.is_calibrating = True
        if not self.__calibration:
            args = dict({'average_height':173})
            frame_wh = frame.shape[1], frame.shape[0]  # float width , height
            self.__calibration = CalibrationYOLO(args, *frame_wh)
            
        res = self.__calibration.extract_entities(frame=frame)
        annotated_frame = res[0].plot()
        
        if self.__calibration.size >= sample_points:
            self.is_calibrating = False
            self.__magic_weight += 1
            self.__magic = self.__calibration.create_lerp_function()
            self.mode = self.__calibration.mode
            self.__calibration = None
            # do the magic func merge dance here probably..
        return annotated_frame
    
    def __find_heads(self, frame):
        if not self.__p2p:
            self.__p2p = PersistentP2P()
        return self.__p2p.process(frame=frame)

    def __create_heatmap(self, points:List[List[float,float]]):
        pass