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
    )


    _app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _app

class Publish(BaseModel):
    channel: str = "Warning"
    message: str


app = get_application()



async def events_ws(websocket:WebSocket):
        await events_ws_sender(websocket),

async def events_ws_reciever(websocket:WebSocket):
        async for message in websocket.iter_text():
            await broadcast.publish(channel="density", message=message)

@app.websocket("/density")
async def events_ws_sender(websocket:WebSocket):
    await websocket.accept()
    async with broadcast.subscribe(channel="density") as subscriber:
            async for event in subscriber:
                await websocket.send_text(event.message)
                await broadcast.publish(channel="density", message="hello world")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/push")
async def push_message(publish:Publish):
    await broadcast.publish(publish.channel, 
    json.dumps(publish.message))
    return Publish(channel= publish.channel,
                   message=json.dumps(publish.message))


@app.websocket("/d")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print("received text")
        try: 
            with open("app/testPictureK.png", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
        except Exception as e:
            print('error', e)
            break
        await websocket.send_json({"image" :encoded_string})

