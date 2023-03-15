from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from app.core.config import settings

class Densities():
    pass

def get_densities():
    _api = FastAPI(title="density")
    _api.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _api

api = get_densities()

@app.get("/data")
async def density():
    temp = pd.read_csv('dummydata.csv')
    data = temp.to_dict() 
    return {"data": data}, 200