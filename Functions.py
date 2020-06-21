from flask import Flask, request, jsonify, abort, Response
import tensorflow
import keras
import json
import base64
import time
import os
from imageai.Detection import ObjectDetection

json_config = json.load(open('config.json'))

def load_model(model_path = json_config['model_file']['default'], detection_speed = json_config['default_parameters']['default_speed']):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel(detection_speed=detection_speed)

    return detector

def Detection(detector, objects, image_path, minimum_probability = json_config['default_parameters']['default_probability']):
    input_dir = json_config['default_parameters']['upload_directory']
    output_dir = json_config['default_parameters']['detected_directory']

    input_path = input_dir + image_path
    output_path = output_dir + image_path

    if len(objects)!=0:
        custom = detector.CustomObjects() 

        for item in objects:
            custom[item]='valid'

        try :
            detections = detector.detectCustomObjectsFromImage(custom_objects = custom, input_image= input_path, output_image_path= output_path,minimum_percentage_probability=minimum_probability, thread_safe=True)
        except :
            abort(Response('Invalid image format', status= 400))
    else :

        try:
            detections = detector.detectObjectsFromImage(input_image= input_path, output_image_path= output_path,minimum_percentage_probability=minimum_probability, thread_safe=True)
        except :
            abort(Response('Invalid image format', status= 400))  

    #delete_image(input_path)
    return detections

def upload_image(imagefile, image_name, image_type):
    image_path = image_name + str(time.time()) + "." + image_type
    try:
        with open(json_config['default_parameters']['upload_directory'] + image_path, "wb") as fh:
            fh.write(base64.b64decode(imagefile))
    except :
        abort(Response('Invalid image', status= 400))
    return image_path

def delete_image(input_path):
    if(os.path.exists(input_path)):
        os.remove(input_path)
         

def handle_request(json_body):
    image_file = json_body.get('image')
    image_name = json_body.get('image_name')
    image_type = json_body.get('image_type')

    objects = json_body.get('objects') if json_body.get('objects')!=None else json_config['default_objects']
    minimum_probability = json_body.get('minimum_probability') if json_body.get('minimum_probability')!=None else json_config['default_parameters']['default_probability']
    detection_speed = json_body.get('detection_speed') if json_body.get('detection_speed')!=None else json_config['default_parameters']['default_speed']

    if not isinstance(image_file, str) or not isinstance(image_name, str) or not isinstance(image_type, str) or not isinstance(objects, list) or not isinstance(minimum_probability, int) and not isinstance(minimum_probability, float) or not isinstance(detection_speed, str) :
        abort(400)

    return image_name, image_type, image_file, objects, minimum_probability, detection_speed
    