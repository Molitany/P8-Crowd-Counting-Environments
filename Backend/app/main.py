import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.core.config import settings
#from starlette.concurrency import run_until_complete 
from broadcaster import Broadcast
from pydantic import BaseModel

import pandas as pd
import asyncio
import time
import base64
broadcast = Broadcast('memory://')

def get_application():
    _app = FastAPI(
        title=settings.PROJECT_NAME,
        on_startup=[broadcast.connect],
        on_shutdown = [broadcast.disconnect]
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
        self.active_connections.append(websocket)

    def disconnect(self,websocket:WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message:str,websocket:WebSocket):
        await websocket.send_text(message)

    async def sendJson(self, message:json,websocket:WebSocket):
        await websocket.send_json(message)

    async def broadcastWarning(self, message:str):
        for connection in self.active_connections:
            await connection.send_json({"warning": message})

    

manager = ConnectionManager()


app = get_application()


@app.get("/")
async def root():
    return {"message": "Hello World"}



@app.websocket("/density")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    while True:
        data = await websocket.receive_text()
        print("received text")
        try: 
            with open("app/testPictureK.png", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                await manager.sendJson({"image" :encoded_string},websocket)
                
        except Exception as e:
            print('error', e)
            break
        await asyncio.sleep(5)
        try: 
            with open("app/testPictureMK.png", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                await manager.sendJson({"image" :encoded_string},websocket)
                
        except Exception as e:
            print('error', e)
            break

        

async def P2P():
    while True:
        print("TESTA")
        await asyncio.  sleep(10)
        print("TESTB")
        await manager.broadcastWarning("TEST")

loop = asyncio.get_event_loop()

if not loop.is_running():
    loop.run_until_complete(P2P())
else: 
    loop.create_task(P2P())