import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.core.config import settings
from concurrent.futures import ThreadPoolExecutor
from PIL import Image as im
import pandas as pd
import asyncio
import base64


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

import cv2
from .FrameProcessing import MagicFrameProcessor
def P2P():
    cap = cv2.VideoCapture(0)
    magic = MagicFrameProcessor()
    # Only for testing purposes remove later
    testCount = 0
    while True:
        success, frame = cap.read()
        if success:
            count, img = magic.process(frame=frame)
            if testCount % 20 == 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = im.fromarray(img)

                # This is terrible and i hate it. 
                image.save('output_image.png')
                with open("output_image.png", "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # Make a new event loop and call this function.
                asyncio.run(manager.broadcastWarning(encoded_string,count))

            cv2.imshow("Video out", img)
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
def start_P2P():
    # Execute the p2p function asynchronously and return a future object
    executor.submit(P2P)
# Start the seperate thread with the P2P function
start_P2P()
    