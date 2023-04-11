import cv2
from ultralytics import YOLO
import argparse
from enum import IntEnum
from typing import List
import numpy as np
import math


def parse_args():
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


def display_magic_curve(magic, height:int):
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    x = [magic(y) for y in range(int(height))]
    plt.plot(x)
    plt.title('Line graph')
    plt.xlabel('image height')
    plt.ylabel('magic number (cm/pixel)')
    plt.show()

def display_lerps(lerps, pred, x):
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    for i in lerps:
        #plt.scatter(x, i, color="black")
        plt.plot(x, i, color="black")
    plt.plot(x, pred, color="blue", linewidth=1)
    plt.xticks(())
    plt.yticks(())
    plt.show()

class BoundingBox:
    def __init__(self, xyxy, wh) -> None:
        self.x1 = xyxy[0]
        self.y1 = xyxy[1]
        self.x2 = xyxy[2]
        self.y2 = xyxy[3]
        self.w = wh[0]
        self.h = wh[1]

class CalibrationYOLO:
    def __init__(self, args, frame_width, frame_height) -> None:
        # Load the YOLOv8 model
        self.model:YOLO = YOLO('yolov8n.pt') # architecture+weights
        self.args = AttributeDict(args) # store system args
        # pred args
        self.dict_args = {
            'stream': False, # as iterator if true
            'classes':Labels.get_all()
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
        def magic(y,p1=p1,p2=p2):
            p1_cm_per_pixel = avg_height/p1.h
            p2_cm_per_pixel = avg_height/p2.h
            p3_cm_per_pixel = p1_cm_per_pixel + (y - p1.y1) * ((p2_cm_per_pixel - p1_cm_per_pixel)/(p2.y1 - p1.y1)) #lerp cm per pixel / scale, probably
            return p3_cm_per_pixel
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
        margin = 0.02 #0.05
        w_margin = self.frame_width * margin
        h_margin = self.frame_height * margin
        
        margin_conditions = [
            (bb.x1 > w_margin),
            (bb.x2 < (self.frame_width - w_margin)),
            (bb.y1 > h_margin),
            (bb.y2 < (self.frame_height - h_margin))
        ]
        size_upper_limit = 0.7
        size_lower_limit = 0.05#0.3
        bbul = self.frame_height * size_upper_limit
        bbll = self.frame_height * size_lower_limit
        size_conditions = [
            (bbul > bb.h > bbll) 
        ]
        proximity_margin = 0.05
        prox = self.frame_height * proximity_margin
        other_y = [x.y1 for x in self.list_of_people]
        # prox is matrix based so substraction is upper and addition is lower.
        proximity_conditions = [(bb.y1 < (oy-prox) or bb.y1 > (oy+prox)) for oy in other_y]
        
        no_square_people_conditions = [
            (bb.h > bb.w * 3.7) # real ratio is 3.9
        ]
        
        conditions = margin_conditions + size_conditions + proximity_conditions + no_square_people_conditions#+ height_conditions
        valid = all(conditions)
        if valid:
            self.list_of_people.append(bb)
        if self.args.test:
            print('BB\'s', len(self.list_of_people))
            print("{} / {} margin_conditions held.".format(sum([1 for x in margin_conditions if x]), len(margin_conditions)))
            print("{} / {} size_conditions held.".format(sum([1 for x in size_conditions if x]), len(size_conditions)))
            print("{} / {} proximity_conditions held.".format(sum([1 for x in proximity_conditions if x]), len(proximity_conditions)))
            print("{} / {} no_square_people_conditions held.".format(sum([1 for x in no_square_people_conditions if x]), len(no_square_people_conditions)))

    def extract_entities(self, frame):
        # Run YOLOv8 inference on the frame
        results = self.model(frame, **self.dict_args)
        for res in results:
            pred_classes = res.boxes.cls.cpu().numpy().tolist()
            pred_classes = [int(x) for x in pred_classes]
            for box, pred_class in zip(res.boxes, pred_classes):
                #print(box)
                box = box.cpu().numpy()
                conf = box.conf.tolist()[0]
                print("Class: {}, Conf {}".format(Labels(pred_class)._name_ , conf))
                if conf < 0.4:
                    continue
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
                    print('>>### PHONE REGISTERED ###<<\n'*5)
        return results

    def create_lerp_function(self):
        """
        * create mega lerp
        * LinearRegression^TM (now np.polyfit)

        returns magic scaling function
        """
        assert len(self.list_of_people) > 1, 'Not enough valid bounding boxes' 
        frame_h_arr = list(range(self.frame_height))
        mega_lerps = []
        bb2s = self.list_of_people[1:] # remove first + copy
        for bb1 in self.list_of_people:
            for bb2 in bb2s:
                mega_lerps.append(self.real_magic(frame_h_arr, bb1, bb2, avg_height=self.args.average_height))
                bb2s.pop(0)   
        mega_lerp = sum(mega_lerps, [])
        lerps = len(mega_lerp) // self.frame_height # interger divide -> amount of 'lines'
        a,b = np.polyfit(lerps * frame_h_arr, mega_lerp, 1)
        magic = lambda x: a*x+b

        if self.args.test:
            display_lerps(mega_lerps, [magic(x) for x in frame_h_arr], frame_h_arr)
        
        return magic

def lerp_engine_stream(stream:cv2.VideoCapture, _args):
    args = _args
    if isinstance(args, dict):
        args = AttributeDict(args) # to allow .dot notation
    frame_wh = stream.get(3), stream.get(4)  # float width , height
    assert args.points and args.average_height, 'Not enough args. fx {"points":8,"average_height":173}'
    assert frame_wh[0] > 0 and frame_wh[1] > 0, "Stopping due to no video capture available."
    mbr = CalibrationYOLO(args, *frame_wh)
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
    magic_func = lerp_engine_stream(cap, args)
    display_magic_curve(magic=magic_func, height=cap.get(4))
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

