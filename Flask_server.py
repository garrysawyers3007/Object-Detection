from flask import *
import tensorflow
import keras
import json
import base64
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
import time
from imageai.Detection import ObjectDetection
import werkzeug

app = Flask(__name__)

detector = ObjectDetection()

@app.route('/image/', methods=['POST'])
def image_request():
    if(request.method=='POST'):

        json_body= request.get_json()
        imagefile = json_body.get('image')
        objects=json_body.get('objects')
        
        with open("imageToSave.png", "wb") as fh:
            fh.write(base64.b64decode(imagefile))

        t0=time.perf_counter()
        if len(objects)!=0:
            custom = detector.CustomObjects()

            for item in objects:
                custom[item]='valid'

            print('Custom')
            detections = detector.detectCustomObjectsFromImage(custom_objects = custom, input_image= 'imageToSave.png', output_image_path= 'imageToSavetest.png',minimum_percentage_probability=35, thread_safe=True)
        else :
            print('Not custom')
            detections = detector.detectObjectsFromImage(input_image= 'imageToSave.png', output_image_path= 'imageToSavetest.png',minimum_percentage_probability=35, thread_safe=True)

        t1=time.perf_counter()
        #detections = json.dumps(detections, default= int, skipkeys=True)

        print("Time elapsed in detection:", t1-t0)
        list = [0,0,0,0], [1,1,1,1]

        return jsonify(
            objects = json.dumps(detections, default= int)
        )
         
@app.route('/', methods=['GET'])
def hello():
    return "Hello"

if __name__ == '__main__':
    t0=time.perf_counter()

    detector.setModelTypeAsYOLOv3()
    detector.setModelPath('yolo.h5')
    detector.loadModel(detection_speed='fastest')
    
    
    t1=time.perf_counter()
    print("Time elapsed in loading model:", t1-t0)
    app.run(debug=True)