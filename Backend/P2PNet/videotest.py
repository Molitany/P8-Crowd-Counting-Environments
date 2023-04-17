import os
import cv2
from app import main
import datetime
from PIL import Image as im

get_path = lambda *x : os.path.join(os.path.dirname(__file__),*x)


TIME = datetime.datetime.now().time()
CROWD_THRESHOLD = 4

def save_frame_range(video_path, start_frame, stop_frame, step_frame,
                     dir_path, basename, ext='png'):
    video = cv2.VideoCapture(video_path) # should be a video stream later

    if not video.isOpened():
        return
   
    os.makedirs(dir_path, exist_ok=True)
   
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(video.get(cv2.CAP_PROP_FRAME_COUNT))))
    for n in range(start_frame, stop_frame, step_frame):
        video.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = video.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
        else:
            return
   
def roundSeconds(dateTimeObject):
    newDateTime = dateTimeObject + datetime.timedelta(seconds=.5)
    return newDateTime.replace(microsecond=0)

if __name__ == '__main__':
   
   print("A")
   save_frame_range(get_path('videofile.mp4'), 0, 750, 750, get_path('images'), 'img_test') 
   print("B")
   
   for filename in os.listdir(get_path('images')): 
        print("C")
        f = os.path.join(get_path('images'), filename)
        print(f)
        print("D")
        if os.path.isfile(f):
            print("E")
            count,img = main(f)
            print("F")

            print(type(img))
            if count > CROWD_THRESHOLD:
                new_img = im.fromarray(img)
                cv2.imwrite(os.path.join("./", 'pred{}.{}'.format(count,str(roundSeconds(datetime.datetime.now()).time()).replace(':', '.'))+'.jpg'.format(count)), img)