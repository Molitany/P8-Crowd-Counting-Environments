try:
    from Calibration import CalibrationYOLO, display_magic_t_curve
    from P2PNet import PersistentP2P
except:
    from . import CalibrationYOLO
    from . import PersistentP2P
from typing import Tuple, List
import numpy as np
import torch
import cv2
import gc
import os
from shapely.ops import unary_union
from shapely.geometry import Point, MultiPolygon
from shapely.geometry.polygon import Polygon
import atexit
get_path = lambda *x: os.path.join(os.path.dirname(__file__), *x); 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if os.path.exists(get_path('bus.jpg')):
    print("WARNING 'source' is found. Removing bus.jpg")
    os.remove(get_path('bus.jpg'))
    
test_vids = {
    'benno': {
        'path': 'videos/benno.mp4',
    },
    'crosswalk_s': {
        'path': 'videos/Crosswalk_s.mp4',
    },
    'crosswalk': {
        'path': 'videos/Crosswalk.mp4',
    },
    'istock': {
        'path': 'videos/istock-962060884_preview.mp4',
    },
    'full': {
        'path': 'videos/12345.mp4',
    },
    '2': {
        'path': 'videos/2.mp4',
    },
    'Experiment_1': {
        'path': 'videos/Experiment_1.mp4',
    },
    'Experiment_2': {
        'path': 'videos/Experiment_2.mp4',
    },
    'Experiment_1_speed': {
        'path': 'videos/Experiment_1_speed.mp4',
    },
    'Experiment_2_speed': {
        'path': 'videos/Experiment_2_speed.mp4',
    },
    'Experiment_2_480p': {
        'path': 'videos/Experiment_2_480p.mp4',
    },
    'Experiment_1_480p': {
        'path': 'videos/Experiment_1_480p.mp4',
    },
    'Expriment_1_480p_no_square_people': {
        'path': 'videos/Expriment_1_480p_no_square_people.mp4',
    },
    'Expriment_2_480p_no_square_people': {
        'path': 'videos/Expriment_2_480p_no_square_people.mp4',
    }
}


class MagicFrameProcessor:
    """
    MagicFrameProcessor^TM - we dont know either.

    Public:

    .is_calibrating
    - bool flag for deducing if in calibration mode or not.

    .current_mode
    - int flag indicating operation mode.

    .process(frame, optional[style])
    - image pipeline entry.
    """

    def __init__(self, test: bool = False, force_height: int = 172, frames_for_recalibration=100, show_metrics=True, continues_recalibration=False) -> None:
        # calibration args
        self.__test = test
        self.__use_human_height = force_height
        self.__show_metrics = show_metrics

        # public by property
        self.__mode = 0  # p2p mode

        # magic scaling function and current weight
        self.__magic = None  # scale func
        self.__magic_weight = 0  # scale func weight for re-calibration
        self.__magic_t = [] # a over time
        self.__magic_weights_t = [] # weight over time

        # recalibration
        self.__continue_recalibration = True # activate recalibration
        self.__frames_for_recalibration = frames_for_recalibration
        self.__recalibration_image_folder = 'recalibration_dataset'
        self.__reclaibration_list_size = 0 # starting size of dir
        self.continues_recalibration = continues_recalibration
        self.__clear_images__in_folder(folder=self.__recalibration_image_folder)
        self.__frame_list = []

        # ML persistance objects
        self.__calibration = None  # calibration obj containing YOLOv8
        self.__p2p = None  # prediction object containing PersistentP2P

        # idk - del function
        atexit.register(self.__close)

    def __close(self):
        if self.__show_metrics:
            print('Magic a/time',self.__magic_t)
            print('Magic weights',self.__magic_weights_t)
            display_magic_t_curve(self.__magic_t)

    @property
    def current_mode(self) -> int:
        """ :returns the operation mode set by the calibration. """
        return self.__mode

    def process(self, frame: np.ndarray, style: int = 2) -> Tuple[bool, int, np.ndarray]:
        """
        This is the main function. It takes an image array and scales the image to a multiple of 128.
        Then if there is no calibration, the image is fed to the YOLO calibration object.
        If there is a calibration, the image is fed to the P2P persistence object, which returns pixel coordinates.
        This is used to create a heatmap.

        :arg frame is an image as a numpy array.
        :arg style is an integer between 1-5. 1=squares, 2=squares+dotted-heads, 3=dotted-heads, 4=overlayed-heatmap, 5=raw-heatmap.
        :return Tuple (alert:bool, count:int, heatmap:numpy.ndarray)

        :throws ValueError on arg style out-of-range.
        """
        width, height = frame.shape[1], frame.shape[0]
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        frame = cv2.resize(frame, dsize=(new_width, new_height),
                           interpolation=cv2.INTER_LANCZOS4)

        if not self.__magic:  # change todo re-calibration
            return False, *self.__calibrate(frame=frame)
        elif self.__continue_recalibration:
            self.__perform_recalibration(frame)
            
            

        count, head_coords = self.__find_heads(frame=frame)
        multi_polygon = self.__create_squares(frame, head_coords)
        alert: bool = multi_polygon.area > 0.1 * (width*height) #TODO de-magic

        if style == 1:
            frame = self.__show_squares(frame, multi_polygon)
        elif style == 2:
            frame = self.__show_squares(frame, multi_polygon)
            frame = self.__show_heads(frame, head_coords)
        elif style == 3:
            frame = self.__show_heads(frame, head_coords)
        elif style == 4:
            frame = self.__create_heatmap(frame, head_coords, overlay=True)
        elif style == 5:
            frame = self.__create_heatmap(frame, head_coords, overlay=False)

        else:
            raise ValueError('Out of range ')

        return alert, count, frame

    def __calibrate(self, frame: np.ndarray, sample_points=6) -> Tuple[int, np.ndarray]:
        """
        Feeds the image to YOLOv8 to use for calibration.
        The returned int is always -1 to relay no heatmap creation.

        :arg frame is an image as a numpy array.
        :arg sample_points is the amount of valid calibration points before stopping.

        :return Tuple (signal:int, frame:numpy.ndarray)
        """
        if not self.__calibration:
            args = dict(
                {'test': self.__test, 'average_height': self.__use_human_height})
            frame_wh = frame.shape[1], frame.shape[0]  # float width , height
            self.__calibration = CalibrationYOLO(args, *frame_wh)

        res = self.__calibration.extract_entities(frame=frame)
        annotated_frame = res[0].plot()

        if self.__calibration.size >= sample_points:
            a, b, weight = self.__calibration.create_lerp_merge()
            self.__magic_weight = weight
            if self.__test:
                print(f'### Function a={a} and b={b}')
            self.__magic_t.append(a)
            self.__magic_weights_t.append(weight)
            self.__magic = lambda x: a*x+b
            self.__mode = self.__calibration.mode
            self.__calibration = None  # remove YOLO from memory
            gc.collect()
            torch.cuda.empty_cache()

            # do the magic func merge dance here probably..
        return -1, annotated_frame

    def __perform_recalibration(self, frame:np.ndarray):
        #self.__reclaibration_list_size += 1
        self.__frame_list.append(frame)
        #cv2.imwrite(get_path(self.__recalibration_image_folder,'recali_{}.png'.format(self.__reclaibration_list_size)), frame)
        
        if len(self.__frame_list) < self.__frames_for_recalibration: # recali if x imgs
            return
        
        #yield cv2.putText(img=frame, text="Starting\nrecalibration..", org=(5,5),fontFace=3, fontScale=3, color=(0,0,255), thickness=5)
        
        # cleanup
        self.__p2p = None # kick from server
        gc.collect()
        torch.cuda.empty_cache()
        #reup
        args = dict({'test': self.__test, 'average_height': self.__use_human_height})
        frame_wh = frame.shape[1], frame.shape[0]  # float width , height
        self.__calibration = CalibrationYOLO(args, *frame_wh)
        #recali
        #for diritem in os.listdir(get_path(self.__recalibration_image_folder)):
        for frame_elem in self.__frame_list:
            #saved_image = cv2.imread(get_path(self.__recalibration_image_folder, diritem))
            res = self.__calibration.extract_entities(frame=frame_elem)
            if self.__test:
                annotated_frame = res[0].plot()
                annotated_frame = cv2.putText(img=annotated_frame, text="Starting recalibration...", org=(5,frame.shape[0]//2),fontFace=3, fontScale=1, color=(0,0,255), thickness=2)
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        try:
            a, b, new_weight = self.__calibration.create_lerp_merge(self.__magic, self.__magic_weight)
            self.__magic_t.append(a)
            self.__magic_weights_t.append(new_weight)
            self.__magic = lambda x: a*x+b
            self.__magic_weight = new_weight
            if self.__test or self.__show_metrics:
                print(f'### New recalibrated function a={a} and b={b} (weight={new_weight})')
        except ValueError as e :
            print(e)
            #input('Continue...?')
        #cleanup
        self.__calibration = None
        gc.collect()
        torch.cuda.empty_cache()
        self.__frame_list.clear()
        # reset / stop recali
        if not self.continues_recalibration: # haha yes logic #BrrrrrrrrrrrrrrrrrrrrainRot
            self.__continue_recalibration = False
        self.__reclaibration_list_size = 0
        self.__clear_images__in_folder(folder=self.__recalibration_image_folder)

        return

    def __clear_images__in_folder(self, folder):
        [os.remove(get_path(folder, file)) for file in os.listdir(get_path(folder)) if file.endswith('.png')] # but good
        #os.system('rm {}'.format(get_path(folder,'*.png'))) # remove all png in recalibration folder.

    def __find_heads(self, frame: np.ndarray):
        """ 
        Initializes and calls the P2P model class to keep the model in memory.
        :args: frame is a image as numpy.ndarray.
        :returns list of list of float containing the pixel coordinates of predicted heads.
        """
        if not self.__p2p:
            self.__p2p = PersistentP2P()
        return self.__p2p.process(frame=frame)

    def __get_trigger_polygons(self, head_chords: List[List[float]]) -> List[Polygon]:
        """
        Find if a polygon contains more than 4 people and add a trapezoid of a 1m^2 in real perspective
        :args: head_chords a list of xy coordinates indicating a detected head
        :returns list of polygons triggering the detection
        """
        trigger_points = []
        for hc in head_chords:
            magic = self.__magic
            cmpp = magic(hc[1])
            # convert cm/pixel to pixel/m
            hc_ppm = 100/cmpp
            upper_point = hc[1]-hc_ppm/2
            upper_local_ppm = 100/magic(upper_point)/2
            lower_point = hc[1]+hc_ppm/2
            lower_local_ppm = 100/magic(lower_point)/2

            poly = Polygon([[hc[0]-upper_local_ppm, upper_point], [hc[0]+upper_local_ppm, upper_point], [
                           hc[0]+lower_local_ppm, lower_point], [hc[0]-lower_local_ppm, lower_point]])
            if not poly.is_valid:
                continue
            density = 0
            for ohc in head_chords:
                p_ohc = Point(*ohc)
                if poly.contains(p_ohc):
                    density += 1

            if density >= 4:
                trigger_points.append(poly)
        return trigger_points

    def __show_heads(self, frame: np.ndarray, points: List[List[float]]):
        img_to_draw = frame
        for p in points:
            img_to_draw = cv2.circle(
                frame, (int(p[0]), int(p[1])), int(2), (255, 0, 255), -1)
        return img_to_draw

    def __create_squares(self, frame, head_coords: List[List[float]]):
        trigger_polygons = self.__get_trigger_polygons(head_coords)
        # take union of polygons so only outer line is shown
        cu = unary_union(trigger_polygons)
        cu:MultiPolygon = cu if type(cu) is MultiPolygon else MultiPolygon([cu])
        return cu

    def __show_squares(self, frame:np.ndarray, multipoly:MultiPolygon) -> np.ndarray:
        # convert list of x and list of y to list of x,y to integer
        def int_coords(x): return np.array(x).round().astype(np.int32)
        if not multipoly.is_empty:
            for geom in multipoly.geoms:
                frame = cv2.polylines(frame, [int_coords(geom.exterior.coords)], 1, (0, 0, 255), 2)
        return frame

    def __create_heatmap(self, frame: np.ndarray, points: List[List[float]], overlay: bool = False) -> np.ndarray:
        # draw the predictions
        img_to_draw = np.zeros(frame.shape, np.uint8)
        magic = self.__magic
        normed = np.linalg.norm([magic(0), magic(frame.shape[0])])
        size = 10
        for p in points:
            color = 1
            m = magic(p[1])
            scaled = 1/m * size
            if scaled >= 1:
                scaled = scaled
            else:
                scaled = 1
                color *= m/normed
            img_to_draw = cv2.circle(
                img_to_draw, (int(p[0]), int(p[1])), int(scaled), (255, 255, 255), -1)
        cv2.imshow("points", img_to_draw)

        blur = cv2.GaussianBlur(img_to_draw, (21, 21), 11)
        heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

        if overlay:
            mask = cv2.inRange(heatmap, np.array([128, 0, 0]), np.array(
                [128, 0, 0]))  # np.array([120,255,128]) blue
            mask = 255-mask
            res = cv2.bitwise_and(heatmap, heatmap, mask=mask)
            res_BGRA = cv2.cvtColor(res, cv2.COLOR_BGR2BGRA)
            alpha = res_BGRA[:, :, 3]
            alpha[np.all(res_BGRA[:, :, 1:3] == (0, 0), 2)] = 0
            # alpha[np.all(res_BGRA[:, :, 0:3] != (0, 0, 0), 2)] = 100

            h1, w1 = res_BGRA.shape[:2]
            h2, w2 = frame.shape[:2]
            img1_pad = np.zeros_like(frame)
            img1_pad = cv2.cvtColor(img1_pad, cv2.COLOR_BGR2BGRA)
            yo = (h2-h1)//2
            xo = (w2-w1)//2
            img1_pad[yo:yo+h1, xo:xo+w1] = res_BGRA

            bgr = img1_pad[:, :, 0:3]
            alpha = img1_pad[:, :, 3]
            alpha = cv2.merge([alpha, alpha, alpha])

            overlay = np.where(alpha == 255, bgr, frame)
            return overlay
        return heatmap
    
USE_TEST_V = 'Expriment_1_480p_no_square_people'

def gogovideo(cap,write_video,force_heigth,video_extra=''):
    tick = 0
    output_video = None
    try:
        while True:
            success, frame = cap.read()
            tick += 1
            if success:
                if tick % tickmod == 0:
                    tick = 0
                    trigger, count, img = magic.process(frame=frame)

                    if write_video:
                        if not output_video:
                            output_video = cv2.VideoWriter(f"output_videos/{USE_TEST_V}_{force_heigth}{video_extra}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (img.shape[1],img.shape[0]))
                        output_video.write(img)
                    # else:
                        # cv2.imshow("YOLOv8 Inference", img)
                        # if cv2.waitKey(1) & 0xFF == ord("q"):
                            # break
            else:
                # Break the loop if the end of the video is reached
                break
    finally:
        cap.release()
        if output_video:
            output_video.release()
            print("successful save")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test = False
    write_video = True
    tickmod = 5
    force_heigth = 184
    recalif = 200
    metrics = True
    keep_recali = True

    cap = cv2.VideoCapture(test_vids[USE_TEST_V]['path'])
    magic = MagicFrameProcessor(test=test, force_height=force_heigth, frames_for_recalibration=recalif, show_metrics=metrics, continues_recalibration=keep_recali)
    
    gogovideo(cap=cap,force_heigth=force_heigth,write_video=False,video_extra='_200f_double')
    #magic.continues_recalibration = False
    #cap = cv2.VideoCapture(test_vids[USE_TEST_V]['path'])
    #gogovideo(cap=cap,write_video=write_video,force_heigth=force_heigth,video_extra='_200f_double')