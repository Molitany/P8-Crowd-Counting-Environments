import base64
import cv2
from app import main

# demonstration of conversion to base64 to send over network and conversion back again for fun
if __name__ == "__main__":
    count, img = main("Crosswalk/frame0.jpg")
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode("utf-8")
    with open("./storage.txt", "w") as f:
        f.write(img_str)
    with open("./storage.txt", "r") as f:
        imgstring = f.read()
    imgdata = base64.b64decode(imgstring)
    with open("img.jpg", "wb") as f:
        f.write(imgdata)