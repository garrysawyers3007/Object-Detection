from flask import Flask, request, jsonify, abort, Response
import tensorflow
import keras
import json
import base64
import time
from datetime import date, datetime
import os
from imageai.Detection import ObjectDetection


def check_dirs(paths):
    for path in paths:
        if(not os.path.isdir(path)):
            os.mkdir(path)
    
def get_config(file_name):
    try:
        json_config = json.load(open(file_name))
        return json_config
    except FileNotFoundError as error:
        abort(500)

def load_model(model_path , detection_speed):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel(detection_speed=detection_speed)

    return detector

def Detection(json_config, detector, objects, image_path, minimum_probability, unique_id):
    curr_date = str(date.today())
    curr_time = str(datetime.now().strftime("%H-%M-%S"))

    detected_dir = json_config['default_parameters']['detected_directory']
    upload_dir = json_config['default_parameters']['upload_directory']
    
    input_dir = upload_dir + curr_date + '/'
    output_dir =  detected_dir + curr_date + '/'
    check_dirs([detected_dir, input_dir, output_dir])

    input_path = input_dir + image_path
    output_path = output_dir + image_path

    if len(objects)!=0:
        custom = detector.CustomObjects() 

        for item in objects:
            custom[item]='valid'

        try :
            detections = detector.detectCustomObjectsFromImage(custom_objects = custom, input_image= input_path, output_image_path= output_path,minimum_percentage_probability=minimum_probability, thread_safe=True)
        except Exception as error:
            logger(error, unique_id)
            abort(Response('Error - {}'.format(error), status= 400))
    else :

        try:
            detections = detector.detectObjectsFromImage(input_image= input_path, output_image_path= output_path,minimum_percentage_probability=minimum_probability, thread_safe=True)
        except Exception as error:
            logger(error, unique_id)
            abort(Response('Error - {}'.format(error), status= 400))  

    #delete_image(input_path)
    return detections

def upload_image(json_config, image_file, image_name, image_type, unique_id):

    curr_date = str(date.today())
    curr_time = str(datetime.now().strftime("%H-%M-%S"))

    uploaded_dir = json_config['default_parameters']['upload_directory']
    input_dir = uploaded_dir + curr_date + '/'
    check_dirs([uploaded_dir, input_dir])
    image_path = image_name + '_' + curr_date + '_' + curr_time + "." + image_type

    try:
        with open(input_dir + image_path, "wb") as fh:
            fh.write(base64.b64decode(image_file))
    except Exception as error:
        logger(error, unique_id)
        abort(Response('Error - {}'.format(error), status= 400))
    return image_path

def delete_image(input_path):
    if(os.path.exists(input_path)):
        os.remove(input_path)
         

def handle_image_request(json_body, unique_id):
    image_file = json_body.get('image')
    image_name = json_body.get('image_name')
    image_type = json_body.get('image_type')


    if not isinstance(image_file, str) or not isinstance(image_name, str) or not isinstance(image_type, str) :
        logger('Error - Invalid Request', unique_id)
        abort(Response('Error - Invalid Request', status= 400))

    return image_name, image_type, image_file


def handle_user_request(json_body, json_config):
    objects = json_body.get('objects') if json_body.get('objects')!=None else json_config['default_objects']
    minimum_probability = json_body.get('minimum_probability') if json_body.get('minimum_probability')!=None else json_config['default_parameters']['default_probability']
    detection_speed = json_body.get('detection_speed') if json_body.get('detection_speed')!=None else json_config['default_parameters']['default_speed']
    unique_id = json_body.get('id')

    if not isinstance(objects, list) or not isinstance(minimum_probability, int) and not isinstance(minimum_probability, float) or not isinstance(detection_speed, str) or not isinstance(unique_id, str) :
        logger('Error - Invalid Request', unique_id)
        abort(Response('Error - Invalid Request', status= 400))

    return objects, minimum_probability, detection_speed, unique_id
    
def logger(error, unique_id):
    with open("error.log","a+") as log:
        curr_date = str(date.today())
        curr_time = str(datetime.now().strftime("%H-%M-%S"))
        log_time = curr_date + '_' + curr_time 
        log.write('\n'+log_time + unique_id + '-' + str(error))