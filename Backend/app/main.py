import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from core.config import settings
from concurrent.futures import ThreadPoolExecutor
from PIL import Image as im
import cv2
from FrameProcessing import MagicFrameProcessor
import uvicorn
import pandas as pd
import asyncio
import base64
from io import BytesIO

def get_application():
    _app = FastAPI(
        title=settings.PROJECT_NAME,
    )


    _app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _app

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self,websocket:WebSocket):
        await websocket.accept()
        print("New connection")
        self.active_connections.append(websocket)

    def disconnect(self,websocket:WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message:str,websocket:WebSocket):
        await websocket.send_text(message)

    async def sendJson(self, message:json,websocket:WebSocket):
        await websocket.send_json(message)

    async def broadcastWarning(self, message:str,count:int):
        try:
            for connection in self.active_connections:
                await connection.send_json({"warning": message,"count" : count})
        except Exception as e:
            print(e)

manager = ConnectionManager()


app = get_application()


@app.get("/")
async def root():
    return {"message": "Hello World"}



@app.websocket("/density")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try: 
        while True:
            data = await websocket.receive_text()
            print("received text")
            
    except WebSocketDisconnect: 
        manager.disconnect(websocket)
        print("Client Disconnected")

def frame_processing_handler():
    cap = cv2.VideoCapture(0)
    magic = MagicFrameProcessor()
    # Only for testing purposes remove later
    testCount = 0
    while True:
        success, frame = cap.read()
        if success:
            alert, count, img = magic.process(frame=frame)
            if testCount % 20 == 0:
                #make sure the image 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #get an image from the numpy array
                image = im.fromarray(img)
                #create a buffer to store the image
                buffer = BytesIO()
                #save the image in the buffer in PNG format
                image.save(buffer,format="PNG")
                #Encode the binary data to a base64 string
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                # Make a new event loop and call this function.
                asyncio.run(manager.broadcastWarning(image_base64,count))

            #cv2.imshow("Video out", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
        testCount+=1

    cap.release()
    cv2.destroyAllWindows()        



executor = ThreadPoolExecutor()
# chedules the function to run in a different thread
def start_frame_processing():
    try:
    # Execute the p2p function asynchronously and return a future object
        executor.submit(frame_processing_handler)
    finally: #currently this does nothing. Hoped to fix the thread not terminating on MacOS.
        executor.shutdown(False)
# Start the seperate thread with the P2P function
start_frame_processing()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)