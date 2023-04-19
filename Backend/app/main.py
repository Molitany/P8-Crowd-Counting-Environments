import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.core.config import settings
from concurrent.futures import ThreadPoolExecutor
import cv2
from . import MagicFrameProcessor

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

    async def broadcastWarning(self, message:str):
        try:
            for connection in self.active_connections:
                await connection.send_json({"warning": message})
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
            try: 
                with open("app/testPictureK.png", "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    await manager.sendJson({"image" :encoded_string},websocket)
            except Exception as e:
                print('error [[', e,']] Websocket will now be closed')
                manager.disconnect(websocket)
                break
            await asyncio.sleep(5)
            try: 
                with open("app/testPictureMK.png", "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    await manager.sendJson({"image" :encoded_string},websocket)
                    
            except Exception as e:
                print('error [[', e,']] websocket will now be closed')
                manager.disconnect(websocket)
                break
    except WebSocketDisconnect: 
        manager.disconnect(websocket)
        print("Client Disconnected")

def P2P2():
    cap = cv2.VideoCapture(0)
    magic = MagicFrameProcessor()
    while True:
        success, frame = cap.read()
        if success:
            count, img = magic.process(frame=frame)

            cv2.imshow("Video out", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    cap.release()
    cv2.destroyAllWindows()        

def P2P():
    while True:
        time.sleep(5)
        #This creates a new event loop to broadcast Warnining, might now be efficent but I don't know enough about event loops to fix it right now
        asyncio.run(manager.broadcastWarning("TEST"))
        print("Sent Warning")

executor = ThreadPoolExecutor()

def start_P2P():
    executor.submit(P2P)

start_P2P()
    
# loop = asyncio.get_event_loop()

# if not loop.is_running():
#     loop.run_until_complete(P2P())
# else: 
#     loop.create_task(P2P())