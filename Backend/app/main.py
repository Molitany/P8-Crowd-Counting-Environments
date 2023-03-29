from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.core.config import settings

import pandas as pd
import time
import base64
def get_application():
    _app = FastAPI(title=settings.PROJECT_NAME)

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _app


app = get_application()

@app.get("/")
async def root():
    return {"message": "Hello World"}




@app.websocket("/density")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print("received text")
        try: 
            with open("app/testPictureK.png", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                print(encoded_string)
        except Exception as e:
            print('error', e)
            break
        await websocket.send_json({"image" :encoded_string})


status = pd.read_csv('app/dummydata.csv').to_dict()

@app.websocket("/densty")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try: 
            await websocket.send_json(status)
            time.sleep(10)
        except Exception as e:
            print('error', e)
            break
    print('bye..\n')


