from cv2 import VideoCapture

class Stream():
    def __init__(self):
        self.camera: VideoCapture = VideoCapture(0)
    
    def show(self):
        self.camera

    
stream = Stream()