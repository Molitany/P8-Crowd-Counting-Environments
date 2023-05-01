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
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from shapely.ops import unary_union
from shapely.geometry import Point, MultiPolygon
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
get_path = lambda *x: os.path.join(os.path.dirname(__file__), *x)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
if os.path.exists(get_path('bus.jpg')):
    print("WARNING 'source' is found. Removing bus.jpg")
    os.remove(get_path('bus.jpg'))
USE_TEST_V = 'crosswalk_s'
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

        # public by property
        self.__mode = 0  # p2p mode

        # magic scaling function and current weight
        self.__magic = lambda x: -0.0036982590391523252*x+7.081969529894982  # scale func
        self.__magic_weight = 1  # scale func weight for re-calibration

        # recalibration
        self.__continue_recalibration = True  # activate recalibration
        self.__recalibration_image_folder = 'recalibration_dataset'
        self.__reclaibration_list_size = 0  # starting size of dir
        self.__clear_images__in_folder(
            folder=self.__recalibration_image_folder)

        # ML persistance objects
        self.__calibration = None  # calibration obj containing YOLOv8
        self.__p2p = None  # prediction object containing PersistentP2P

    @property
    def current_mode(self) -> int:
        """ :returns the operation mode set by the calibration. """
        return self.__mode

    def process(self, frame: np.ndarray, style: int = 5) -> Tuple[bool, int, np.ndarray]:
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
        alert: bool = multi_polygon.area > 0.1 * (width*height) # TODO de-magic
        self.__Gaussian_density(frame, head_coords)
        # if style == 1:
        #     frame = self.__show_squares(frame, multi_polygon)
        # elif style == 2:
        #     frame = self.__show_squares(frame, multi_polygon)
        #     frame = self.__show_heads(frame, head_coords)
        # elif style == 3:
        #     frame = self.__show_heads(frame, head_coords)
        # elif style == 4:
        #     frame = self.__KDE(frame, head_coords)
        # elif style == 5:
        #     frame = self.__create_heatmap(frame, head_coords, overlay=True)
        #     frame = self.__create_heatmap(frame, head_coords, overlay=False)

        # else:
        #     raise ValueError('Out of range ')

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
            self.__magic = lambda x: a*x+b
            self.__mode = self.__calibration.mode
            self.__calibration = None  # remove YOLO from memory
            gc.collect()
            torch.cuda.empty_cache()

            # do the magic func merge dance here probably..
        return -1, annotated_frame

    def __perform_recalibration(self, frame: np.ndarray):
        self.__reclaibration_list_size += 1
        cv2.imwrite(get_path(self.__recalibration_image_folder, 'recali_{}.png'.format(
            self.__reclaibration_list_size)), frame)

        if self.__reclaibration_list_size < 100:  # recali if 10 imgs
            return

        # yield cv2.putText(img=frame, text="Starting\nrecalibration..", org=(5,5),fontFace=3, fontScale=3, color=(0,0,255), thickness=5)

        # cleanup
        self.__p2p = None  # kick from server
        gc.collect()
        torch.cuda.empty_cache()
        # reup
        args = dict(
            {'test': self.__test, 'average_height': self.__use_human_height})
        frame_wh = frame.shape[1], frame.shape[0]  # float width , height
        self.__calibration = CalibrationYOLO(args, *frame_wh)
        # recali
        for diritem in os.listdir(get_path(self.__recalibration_image_folder)):
            saved_image = cv2.imread(
                get_path(self.__recalibration_image_folder, diritem))
            res = self.__calibration.extract_entities(frame=saved_image)
            if self.__test:
                annotated_frame = res[0].plot()
                annotated_frame = cv2.putText(img=annotated_frame, text="Starting recalibration...", org=(
                    5, frame.shape[0]//2), fontFace=3, fontScale=1, color=(0, 0, 255), thickness=2)
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        a, b, new_weight = self.__calibration.create_lerp_merge(
            self.__magic, self.__magic_weight)
        self.__magic = lambda x: a*x+b
        self.__magic_weight = new_weight

        # cleanup
        self.__calibration = None
        gc.collect()
        torch.cuda.empty_cache()
        if self.__test:
            print(
                f'### New recalibrated function a={a} and b={b} (weight={new_weight})')
        # reset / stop recali
        self.__continue_recalibration = False
        self.__reclaibration_list_size = 0
        self.__clear_images__in_folder(
            folder=self.__recalibration_image_folder)

        return

    def __clear_images__in_folder(self, folder):
        [os.remove(get_path(folder, file)) for file in os.listdir(
            get_path(folder)) if file.endswith('.png')]  # but good
        # os.system('rm {}'.format(get_path(folder,'*.png'))) # remove all png in recalibration folder.

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
            density = 0
            for ohc in head_chords:
                p_ohc = Point(*ohc)
                if poly.contains(p_ohc):
                    density += 1

            if density >= 4:
                trigger_points.append(poly)
        return trigger_points

    def __show_heads(self, frame: np.ndarray, points: List[List[float]]):
        for p in points:
            img_to_draw = cv2.circle(
                frame, (int(p[0]), int(p[1])), int(2), (0, 0, 255), -1)
        return img_to_draw

    def __create_squares(self, frame, head_coords: List[List[float]]):
        trigger_polygons = self.__get_trigger_polygons(head_coords)
        # take union of polygons so only outer line is shown
        cu = unary_union(trigger_polygons)
        cu: MultiPolygon = cu if type(
            cu) is MultiPolygon else MultiPolygon([cu])
        return cu

    def __show_squares(self, frame: np.ndarray, multipoly: MultiPolygon) -> np.ndarray:
        # convert list of x and list of y to list of x,y to integer
        def int_coords(x): return np.array(x).round().astype(np.int32)
        if not multipoly.is_empty:
            for geom in multipoly.geoms:
                frame = cv2.polylines(
                    frame, [int_coords(geom.exterior.coords)], 1, (0, 0, 255), 2)
        return frame

    # def __create_heatmap(self, frame: np.ndarray, points: List[List[float]], overlay: bool = False) -> np.ndarray:
    #     """
    #     DEPRECATED
    #     """
    #     # draw the predictions
    #     img_to_draw = np.zeros(frame.shape, np.uint8)
    #     magic = self.__magic
    #     normed = np.linalg.norm([magic(0), magic(frame.shape[0])])
    #     size = 10
    #     for p in points:
    #         color = 1
    #         m = magic(p[1])
    #         scaled = 1/m * size
    #         if scaled >= 1:
    #             scaled = scaled
    #         else:
    #             scaled = 1
    #             color *= m/normed
    #         img_to_draw = cv2.circle(
    #             img_to_draw, (int(p[0]), int(p[1])), int(scaled), (255, 255, 255), -1)
    #     cv2.imshow("points", img_to_draw)

    #     blur = cv2.GaussianBlur(img_to_draw, (21, 21), 11)
    #     heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

    #     if overlay:
    #         mask = cv2.inRange(heatmap, np.array([128, 0, 0]), np.array(
    #             [128, 0, 0]))  # np.array([120,255,128]) blue
    #         mask = 255-mask
    #         res = cv2.bitwise_and(heatmap, heatmap, mask=mask)
    #         res_BGRA = cv2.cvtColor(res, cv2.COLOR_BGR2BGRA)
    #         alpha = res_BGRA[:, :, 3]
    #         alpha[np.all(res_BGRA[:, :, 1:3] == (0, 0), 2)] = 0
    #         # alpha[np.all(res_BGRA[:, :, 0:3] != (0, 0, 0), 2)] = 100

    #         h1, w1 = res_BGRA.shape[:2]
    #         h2, w2 = frame.shape[:2]
    #         img1_pad = np.zeros_like(frame)
    #         img1_pad = cv2.cvtColor(img1_pad, cv2.COLOR_BGR2BGRA)
    #         yo = (h2-h1)//2
    #         xo = (w2-w1)//2
    #         img1_pad[yo:yo+h1, xo:xo+w1] = res_BGRA

    #         bgr = img1_pad[:, :, 0:3]
    #         alpha = img1_pad[:, :, 3]
    #         alpha = cv2.merge([alpha, alpha, alpha])

    #         overlay = np.where(alpha == 255, bgr, frame)
    #         return overlay
    #     return heatmap

    def __create_heatmap(self, frame: np.ndarray, points: List[List[float]], overlay: bool = False) -> np.ndarray:
        # draw the predictions
        img_to_draw = np.zeros(frame.shape, np.uint8)
        magic = self.__magic
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), int(1), (255, 255, 255), -1)
        
        blur = cv2.GaussianBlur(img_to_draw, (5,5), -1)
        cv2.imshow("points", blur)
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


    def __Gaussian_density(self, frame, head_coords):
        def __superimpose_matrix(A, B, y, x):
            height, width = frame.shape[:2]
            B_shape = B.shape[0]
            new_shape = B.shape
            if width - x < B_shape:
                new_shape[1] = width - x

            elif width - x > width - B_shape:
                new_shape[1] = -x

            if height - y < B_shape:
                new_shape[0] = height - y
            elif height - y > height - B_shape:
                new_shape[0] = -y

            # TODO: rest of the owl / håndtere out of bounds
            A[y:y+B.shape[0], x:x+B.shape[1]] += B
            return A
        
        def __Gaussian_filter(point, sigma=None, muu=0):
            """
            sigma(standard deviation) and muu(mean) are the parameters of gaussian
            Initializing value of x,y as grid of kernel size
            in the range of kernel size
            """
            magic = self.__magic
            ppm = 100/magic(point[1])
            kernel_size = int(ppm)+1 if int(ppm) % 2 == 0 else int(ppm) # linux kernel must be uneven

            sigma = 1
            x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                            np.linspace(-1, 1, kernel_size))
            dst = np.sqrt(x**2+y**2)

            # lower normal part of gaussian
            normal = 1/(sigma*np.sqrt(2*np.pi))
        
            # Calculating Gaussian filter
            gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal

            middle_index = np.array(gauss.shape)/2
            middle_ratio = 1/gauss[int(middle_index[0])][int(middle_index[1])]
            gauss *= middle_ratio
            return gauss
        
        img = np.zeros(frame.shape[:2])
        for p in head_coords:
            gauss = __Gaussian_filter(p)
            y, x = np.array(gauss.shape)/2
            img = __superimpose_matrix(img, gauss, int(p[1]-y), int(p[0]-x))
        print(img)

    # def __KDE(self, frame: np.ndarray, head_coords: List[List[float]]):
    #     gen = zip(*head_coords)
    #     ys = next(gen)
    #     xs = next(gen)
    #     points = np.vstack([xs, ys])
    #     xmin = np.min(xs)
    #     xmax = np.max(xs)
    #     ymin = np.min(ys)
    #     ymax = np.max(ys)
    #     X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    #     positions = np.vstack([X.ravel(), Y.ravel()])
    #     kde = gaussian_kde(points)
    #     Z = np.reshape(kde(positions), X.shape)

    #     fig, ax = plt.subplots()
    #     ax.imshow(Z, cmap=plt.cm.gist_earth_r,
    #               extent=[0, frame.shape[1], 0, frame.shape[0]])
    #     ax.plot(ys, xs, 'k.', markersize=2)
    #     ax.set_xlim([0, frame.shape[1]])
    #     ax.set_ylim([0, frame.shape[0]])
    #     frame = self.__show_heads(frame, head_coords)
    #     plt.show()
    #     return frame 


if __name__ == '__main__':
    cap = cv2.VideoCapture(test_vids[USE_TEST_V]['path'])
    magic = MagicFrameProcessor(test=True)
    tick = 0
    while True:
        success, frame = cap.read()
        tick += 1
        if success:
            if tick % 1 == 0:
                trigger, count, img = magic.process(frame=frame)

                cv2.imshow("YOLOv8 Inference", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # Break the loop if the end of the video is reached
            break
    cap.release()
    cv2.destroyAllWindows()
