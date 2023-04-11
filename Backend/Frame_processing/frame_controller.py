import cv2
import numpy as np
from Calibration import CalibrationYOLO

class MagicFrameProcessor:
    def __init__(self) -> None:
        self.is_calibrating = False
        self.__calibration = None # calibration obj containing YOLOv8
        self.__magic = None # scale func
        self.__magic_weight = 0 # scale func weight for re-calibration

    def process(self, frame:np.ndarray) -> np.ndarray:
        """
        if not calibrated -> calibrate
         :store magic_func :return none
        else -> p2p -> heatmap
         :store sampling :return heatmap
        """
        if not self.__magic: # change todo re-calibration
            return self.__calibrate(frame=frame)
        
        head_coords = self.__find_heads(frame=frame)
        heatmap = self.__create_heatmap(head_coords)
        return heatmap    

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
            # do the magic func merge dance here probably..
        return annotated_frame
    
    def __find_heads(self, frame):
        pass
