from fastapi import FastAPI, WebSocket
import pandas as pd
import time

from app.core.config import settings

print(pd.read_csv('dummydata').to_dict().count[0])

app = FastAPI(title='Warning system')

@app.websocket("/density")
async def count(websocket: WebSocket):
    print('Accepting client connection....')
    await websocket.accept()
    while True:
        try:    
            status = pd.read_csv('dummydata').to_dict()
            await websocket.send_json(status)
            time.sleep(1)
        except Exception as e:
            print('error', e)
            break
    print('bye..\n')
    