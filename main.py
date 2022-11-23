from typing import Union
from fastapi import FastAPI, UploadFile, Form
import cv2
import easyocr
import numpy as np
import sys
from major_category import *
from major_cosin import *

sys.path.append('.')

reader = easyocr.Reader(['ko'], gpu=False)

def id_ocr(img):
    result = reader.readtext(img, detail=0)
    for x in result:
      if ('대학' in  x) | ('학과' in  x):
        major = x
        break
    
    return major
   
#################################################################################

app = FastAPI()

@app.get("/")
def start():
  return { "hello" : "000000000000000000000000000000000000"}

@app.post("/uploadIDCardImage/")
async def create_upload_file(file: UploadFile):
    file_buf = await file.read()

    encoded_img = np.fromstring(file_buf, dtype = np.uint8)
    image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    major = id_ocr(image)
    major_list = get_recommendations(major)
    result = category(major_list)

    return {"major": result}

