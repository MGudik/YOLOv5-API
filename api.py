import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from typing import Optional, List

from static.render import render
from utilities.environment import Environment
from utilities.logging.config import (initialize_logging,
                                      initialize_logging_middleware)
from utilities.utilities import get_uptime
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import Response, FileResponse

from fastapi.templating import Jinja2Templates
from detector import detect_image
import torch
import numpy as np
import cv2

# --- Welcome to your Emily API! --- #
# See the README for guides on how to test it.

# Your API endpoints under http://yourdomain/api/...
# are accessible from any origin by default.
# Make sure to restrict access below to origins you
# trust before deploying your API to production.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

num_requests = 0

templates = Jinja2Templates(directory=".")

app = FastAPI()

initialize_logging()
initialize_logging_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get('/api')
def hello():
    return {
        "service": Environment().COMPOSE_PROJECT_NAME,
        "uptime": get_uptime(),
        "requests": num_requests
    }

@app.get('/index', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/image')
async def upload(file = File(...), allowed_names: Optional[List[str]] = None):
    contents = await file.read()

    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    detect_image(img, model, allowed_names)

    global num_requests
    num_requests += 1
       

    return FileResponse('temp.png')

@app.post('/image_raw')
async def upload(file: UploadFile = File(...), allowed_names: Optional[List[str]] = None):
    contents = await file.read()

    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detect_image(img, model, allowed_names)

    global num_requests
    num_requests += 1

    return FileResponse('temp.png')


@app.get('/')
def index():
    return HTMLResponse(
        render(
            'static/index.html',
            host=Environment().HOST_IP,
            port=Environment().CONTAINER_PORT
        )
    )


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=Environment().HOST_IP,
        port=Environment().CONTAINER_PORT
    )
