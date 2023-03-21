from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

import pandas as pd
import time


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
        try: 
            status = pd.read_csv('app/dummydata.csv').to_dict()
            await websocket.send_json(status)
            time.sleep(10)
        except Exception as e:
            print('error', e)
            break
    print('bye..\n')


