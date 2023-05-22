from scipy import interpolate
from ultralytics import YOLO
from enum import IntEnum
from typing import List, Optional
import numpy as np
import argparse
import math
import cv2


def parse_args():
    """ args used for testing YOLO directly. """
    parser = argparse.ArgumentParser(description='Calibrate some cams')
    parser.add_argument('-t','--test', action='store_true', help='prints stuff')
    parser.add_argument('-p','--points', type=int, default=10, help='Interger for amount of calibration points')
    parser.add_argument('-a','--average_height', type=int, default=173, help='Interger for estimated average human height in cm')
    args = parser.parse_args()
    print(args)
    return args


class AttributeDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Labels(IntEnum):
    PERSON = 0
    GIRAFFE = 23
    CAKE = 55
    PHONE = 67

    @classmethod
    def get_all(cls):
        return [x.value for x in list(cls)]

def display_magic_t_curve(list_a):
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    x = list_a
    plt.plot(x)
    plt.title('Slope of lerp over time')
    plt.xlabel('Time')
    plt.ylabel('a')
    plt.show()

def display_magic_curve(magic, height:int):
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    x = [magic(y) for y in range(int(height))]
    plt.plot(x)
    plt.title('Final lerp')
    plt.xlabel('Image height (pixel)')
    plt.ylabel('Inverse linear relationship (cm/pixel)')
    plt.show()

def display_lerps(lerps, pred, x):
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    for i in lerps:
        #plt.scatter(x, i, color="black")
        plt.plot(x, i, color="black")
    plt.title('All lerps')
    plt.xlabel('Image height (pixel)')
    plt.ylabel('Inverse linear relationship (cm/pixel)')
    plt.plot(x, pred, color="blue", linewidth=1)
    plt.show()

class BoundingBox:
    def __init__(self, xyxy, wh) -> None:
        self.x1 = xyxy[0]
        self.y1 = xyxy[1]
        self.x2 = xyxy[2]
        self.y2 = xyxy[3]
        self.w = wh[0]
        self.h = wh[1]
    
    def __repr__(self) -> str:
        return str(hash((self.x1,self.y1,self.x2,self.y2,self.w,self.h)))

class CalibrationYOLO:
    def __init__(self, args, frame_width, frame_height) -> None:
        # Load the YOLOv8 model
        self.model:YOLO = YOLO('yolov8n.pt') # architecture+weights
        self.args = AttributeDict(args) # store system args
        # pred args
        self.dict_args = {
            'stream': False, # as iterator if true
            'classes':Labels.get_all(),
            'max_det':1200,
            'conf':0.30,
            'verbose': False
        }
        self.frame_height = math.ceil(frame_height)
        self.frame_width = math.ceil(frame_width)
        self.list_of_people:List[BoundingBox] = []
        self.magic_mode:int = 0


    @property
    def size(self) -> int:
        return len(self.list_of_people)

    @property
    def mode(self) -> int:
        return self.magic_mode

    def real_magic(self, yrow: List | int, p1:BoundingBox, p2:BoundingBox,avg_height=173):
        p1_cm_per_pixel = avg_height/p1.h
        p2_cm_per_pixel = avg_height/p2.h
        x = [p1.y1, p2.y1]
        y = [p1_cm_per_pixel, p2_cm_per_pixel]
        magic = interpolate.interp1d(x, y, fill_value='extrapolate')
        if not isinstance(yrow, list):
            yrow = range(yrow)
        res = [magic(x) for x in yrow]
        return res

    def bb_add_if_valid(self, bb:BoundingBox):
        """
        no bb in margin 
        no far far away 
        no too too big 
        no y-axis overlap 
        no same bb height 
        """
        margin = 0.15 #0.05
        w_margin = self.frame_width * margin
        h_margin = self.frame_height * margin
        
        margin_conditions = [
            (bb.x1 > w_margin),
            (bb.x2 < (self.frame_width - w_margin)),
            (bb.y1 > h_margin),
            (bb.y2 < (self.frame_height - h_margin))
        ]
        proximity_margin = 0.03
        prox = self.frame_height * proximity_margin
        other_y = [x.y1 for x in self.list_of_people]
        # prox is matrix based so substraction is upper and addition is lower.
        proximity_conditions = [(bb.y1 < (oy-prox) or bb.y1 > (oy+prox)) for oy in other_y]
                
        conditions = margin_conditions + proximity_conditions
        valid = all(conditions)

        if valid:
            self.list_of_people.append(bb)

        if self.args.test:
            print('BB\'s', len(self.list_of_people))
            print("{} / {} margin_conditions held.".format(sum([1 for x in margin_conditions if x]), len(margin_conditions)))
            print("{} / {} proximity_conditions held.".format(sum([1 for x in proximity_conditions if x]), len(proximity_conditions)))

    def extract_entities(self, frame):
        # Run YOLOv8 inference on the frame
        results = self.model(frame, **self.dict_args)
        for res in results:
            pred_classes = res.boxes.cls.cpu().numpy().tolist()
            pred_classes = [int(x) for x in pred_classes]
            for box, pred_class in zip(res.boxes, pred_classes):
                box = box.cpu().numpy()
                if pred_class == Labels.PERSON:
                    xywh = box.xywh.tolist()[0]
                    xyxy = box.xyxy.tolist()[0]
                    bb = BoundingBox(xyxy, xywh[2:])
                    self.bb_add_if_valid(bb)
                elif pred_class == Labels.CAKE:
                    self.magic_mode = 2
                    print('CAKE '*1000)
                elif pred_class == Labels.PHONE:
                    self.magic_mode = 1
                    print('>>### PHONE REGISTERED ###<<\n'*15)
        return results

    def old_create_lerp_function(self):
        """
        *** DEPRECATED ***
        """
        if not len(self.list_of_people) > 1:
            raise ValueError('Not enough valid bounding boxes')
 
        frame_h_arr = list(range(self.frame_height))
        mega_lerps = []
        bb2s = self.list_of_people[1:] # remove first + copy
        for bb1 in self.list_of_people:
            for bb2 in bb2s:
                line = self.real_magic(frame_h_arr, bb1, bb2, avg_height=self.args.average_height)
                bb2s.pop(0)
                if not line[0] > line[-1]:
                    if self.args.test:
                        print('## Removed outlier produced by:', bb1, '-->', bb2)
                    continue
                mega_lerps.append(line)


        if not len(mega_lerps) > 0:
            raise ValueError('Not enough valid lines found')

        mega_lerp = sum(mega_lerps, [])
        a,b = np.polyfit(len(mega_lerps) * frame_h_arr, mega_lerp, 1)
        magic = lambda x: a*x+b
        weight = len(mega_lerps)
        if self.args.test:
            display_lerps(mega_lerps, [magic(x) for x in frame_h_arr], frame_h_arr)
            display_magic_curve(magic=magic, height=self.frame_height)

        return a,b,weight


    def create_lerp_merge(self, magic=None, weight:Optional[int]=None):
        """
        *** NOT DEPRECATED YET ***
        """
        if not len(self.list_of_people) > 1:
            raise ValueError('Not enough valid bounding boxes')
 
        frame_h_arr = list(range(self.frame_height))
        mega_lerps = []
        bb2s = self.list_of_people[1:] # remove first + copy
        for bb1 in self.list_of_people:
            for bb2 in bb2s:
                line = self.real_magic(frame_h_arr, bb1, bb2, avg_height=self.args.average_height)
                bb2s.pop(0)
                if not line[0] > line[-1]:
                    if self.args.test:
                        print('## Removed outlier produced by:', bb1, '-->', bb2)
                    continue
                mega_lerps.append(line)


        if not len(mega_lerps) > 0:
            raise ValueError('Not enough valid lines found')

        new_weight = weight or len(mega_lerps)

        #incase of recalibration
        if magic:
            #mega_lerps.extend(new_weight*[[magic(x) for x in frame_h_arr]])
            new_weight += len(mega_lerps)
            for __ in range(1, weight or 2):
                m = [magic(x) for x in frame_h_arr]
                mega_lerps.append(m)

        mega_lerp = sum(mega_lerps, [])
        a,b = np.polyfit(len(mega_lerps) * frame_h_arr, mega_lerp, 1)
        if self.args.test:
            magic = lambda x: a*x+b
            display_lerps(mega_lerps, [magic(x) for x in frame_h_arr], frame_h_arr)
            display_magic_curve(magic=magic, height=self.frame_height)

        return a, b, new_weight

def lerp_engine_stream(stream:cv2.VideoCapture, _args):
    """  only used for testing """
    args = _args
    if isinstance(args, dict):
        args = AttributeDict(args) # to allow .dot notation
    frame_wh = stream.get(3), stream.get(4)  # float width , height
    
    if not args.points and not args.average_height:
        raise KeyError('Not enough args. fx {"points":8,"average_height":173}')

    if not frame_wh[0] > 0 and not frame_wh[1] > 0:
        raise ValueError("Stopping due to no video capture available.")

    mbr = CalibrationYOLO(args, *frame_wh)
    print(mbr)

    while True: # calibrating
        # check if done calibrating
        if mbr.size >= args.points:
            break # done calibrating 
        
        success, frame = stream.read()
        if success:
            results = mbr.extract_entities(frame=frame)
            
            if args.test:
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # Break the loop if the end of the video is reached
            break
        
    return mbr.create_lerp_function()


if __name__ == '__main__':
    args = parse_args()
    cap = cv2.VideoCapture('Crosswalk.mp4')
    a,b = lerp_engine_stream(cap, args)
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

