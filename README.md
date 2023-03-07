# P8-Crowd-Counting-Environments
Crowd counting to detect dangerous environments

## Frontend
The frontend is written in React Native using typescript.

To run the frontend:
1. Run `npm install` in the frontend folder.
2. Install [Expo Go](https://expo.dev/expo-go). 
3. Start the frontend with `npm start` and scan the QR code with the app. 

Changes happen live if fast reload is enabled.

## Backend
The backend is written in python3.10, as pytorch does not support 3.11 yet, using FastAPI as a web framework.
You will need to download pytorch yourself as the requirements are dependent on if you have a CUDA card or not and what OS is used. It can be found [here](https://pytorch.org/get-started/locally/).

The database is setup to be postgres with SQLAlchemy as the ORM.

To run the backend:
1. Install the packages with `pip install -r requirements.txt`.
2. Setup a .env file in the Backend folder.
3. Start the web framework with `fastapi run` and go to [localhost:8000](localhost:8000).
