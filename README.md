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
2. Take the env file from [discord](https://discord.com/channels/1070635395731165184/1070636486040506439) and put it in the backend folder.
3. Start the web framework with `fastapi run` and go to [localhost:8000](localhost:8000).
4. P2PNet requires a vgg model in the backend folder to work default is [vgg16_bn](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth) others are available at:
    - [vgg11](https://download.pytorch.org/models/vgg11-bbd30ac9.pth)  
    - [vgg13](https://download.pytorch.org/models/vgg13-c768596a.pth)  
    - [vgg16](https://download.pytorch.org/models/vgg16-397923af.pth)  
    - [vgg19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)  
    - [vgg11_bn](https://download.pytorch.org/models/vgg11_bn-6002323d.pth)  
    - [vgg13_bn](https://download.pytorch.org/models/vgg13_bn-abd245e5.pth)  
    - [vgg16_bn](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth)  
    - [vgg19_bn](https://download.pytorch.org/models/vgg19_bn-c79401a0.pth)