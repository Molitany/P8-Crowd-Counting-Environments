try:
    from Calibration import CalibrationYOLO
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
os.environ['CUDA_VISIBLE_DEVICES'] = ''

USE_TEST_V = 'crosswalk_s'
test_vids = {
    'benno': {
        'path': 'benno.mp4',
    },
    'crosswalk_s': {
        'path': 'Crosswalk_s.mp4',
    },
    'crosswalk': {
        'path': 'Crosswalk.mp4',
    },
    'istock': {
        'path': 'istock-962060884_preview.mp4',
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

    def __init__(self, test: bool = False, force_height: int = 172) -> None:
        # calibration args
        self.__test = test
        self.__use_human_height = force_height

        # public
        self.__is_calibrating = False
        self.__mode = 0  # p2p mode

        # magic scaling function and current weight
        self.__magic = None  # scale func
        self.__magic_weight = 0  # scale func weight for re-calibration

        # ML persistance objects
        self.__calibration = None  # calibration obj containing YOLOv8
        self.__p2p = None  # prediction object containing PersistentP2P

    @property
    def is_calibrating(self) -> bool:
        """ :returns True if the initial calibration is not done. """
        return self.__is_calibrating

    @property
    def current_mode(self) -> int:
        """ :returns the operation mode set by the calibration. """
        return self.__mode

    def process(self, frame: np.ndarray, style: int = 1) -> Tuple[bool, int, np.ndarray]:
        """
        This is the main function. It takes an image array and scales the image to a multiple of 128.
        Then if there is no calibration, the image is fed to the YOLO calibration object.
        If there is a calibration, the image is fed to the P2P persistence object, which returns pixel coordinates.
        This is used to create a heatmap.

        :arg frame is an image as a numpy array.
        :arg style is an integer between 1-3. 1=overlayed-heatmap, 2=raw-heatmap, 3=dotted-heads.
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

        count, head_coords = self.__find_heads(frame=frame)

        trigger_polygons = self.__get_trigger_polygons(head_coords)
        # take union of polygons so only outer line is shown
        cu = unary_union(trigger_polygons)
        # convert list of x and list of y to list of x,y to integer
        def int_coords(x): return np.array(x).round().astype(np.int32)
        if not cu.is_empty:
            if type(cu) is MultiPolygon:
                for geom in cu.geoms:
                    frame = cv2.polylines(
                        frame, [int_coords(geom.exterior.coords)], 1, (0, 0, 255), 4)
            else:
                frame = cv2.polylines(
                    frame, [int_coords(cu.exterior.coords)], 1, (0, 0, 255), 4)
        # if style == 1:
        #     heatmap = self.__create_heatmap(frame, head_coords, overlay=True)
        # elif style == 2:
        #     heatmap = self.__create_heatmap(frame, head_coords, overlay=False)
        # elif style == 3:
        #     heatmap = self.__show_heads(frame, head_coords)
        # else:
        #     raise ValueError('Out of range ')

        alert: bool = False
        return alert, count, frame

    def __calibrate(self, frame: np.ndarray, sample_points=6) -> Tuple[int, np.ndarray]:
        """
        Feeds the image to YOLOv8 to use for calibration.
        The returned int is always -1 to relay no heatmap creation.

        :arg frame is an image as a numpy array.
        :arg sample_points is the amount of valid calibration points before stopping.

        :return Tuple (signal:int, frame:numpy.ndarray)
        """
        self.__is_calibrating = True
        if not self.__calibration:
            args = dict(
                {'test': self.__test, 'average_height': self.__use_human_height})
            frame_wh = frame.shape[1], frame.shape[0]  # float width , height
            self.__calibration = CalibrationYOLO(args, *frame_wh)

        res = self.__calibration.extract_entities(frame=frame)
        annotated_frame = res[0].plot()

        if self.__calibration.size >= sample_points:
            self.__is_calibrating = False
            self.__magic_weight += 1
            a, b = self.__calibration.create_lerp_function()
            if self.__test:
                print(f'### Function a={a} and b={b}')
            self.__magic = lambda x: a*x+b
            self.__mode = self.__calibration.mode
            self.__calibration = None  # remove YOLO from memory
            gc.collect()
            torch.cuda.empty_cache()

            # do the magic func merge dance here probably..
        return -1, annotated_frame

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
            ppm = 100/cmpp  # convert cm/pixel to pixel/m
            poly = Polygon([[hc[0]-ppm*0.75, hc[1]-ppm], [hc[0]+ppm*0.75, hc[1]-ppm], [
                           hc[0]+ppm*0.75*2, hc[1]+ppm], [hc[0]-ppm*0.75*2, hc[1]+ppm]])
            density = 0
            for ohc in head_chords:
                p_ohc = Point(*ohc)
                if poly.contains(p_ohc):
                    density += 1

            if density > 4:
                trigger_points.append(poly)
        return trigger_points

    def __show_heads(self, frame: np.ndarray, points: List[List[float]]):
        for p in points:
            img_to_draw = cv2.circle(
                frame, (int(p[0]), int(p[1])), int(2), (0, 0, 255), -1)
        return img_to_draw

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


if __name__ == '__main__':
    cap = cv2.VideoCapture(test_vids[USE_TEST_V]['path'])
    magic = MagicFrameProcessor(test=False)
    tick = 0
    while True:
        success, frame = cap.read()
        tick += 1
        if success:
            if tick % 3 == 0:
                tick = 0
                trigger, count, img = magic.process(frame=frame)

                cv2.imshow("YOLOv8 Inference", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # Break the loop if the end of the video is reached
            break
    cap.release()
    cv2.destroyAllWindows()
