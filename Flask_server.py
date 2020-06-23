from Functions import *

app = Flask(__name__)

detector = None 
json_config = None

@app.route('/image/', methods=['POST'])
def image_request():
    if(request.method=='POST'):

        json_body= request.get_json() #get json from request

        #get data from the request json
        objects, minimum_probability, detection_speed, unique_id = handle_user_request(json_body, json_config)

        image_name, image_type, image_file = handle_image_request(json_body, unique_id)

        #upload the retrieved image to the uploads directory 
        image_path = upload_image(json_config, image_file, image_name, image_type, unique_id)

        # detect the objects in the image as per user input
        t0=time.perf_counter()
        detections = Detection(json_config, detector, objects, image_path, minimum_probability, unique_id)
        t1=time.perf_counter()
        
        print("Time elapsed in detection:", t1-t0)

        #return the response
        return jsonify(
            objects = json.dumps(detections, default= int)
        )
         
@app.route('/', methods=['GET'])
def hello():
    return "Hello"

if __name__ == '__main__':

    json_config = get_config('config.json')

    upload_path = json_config['default_parameters']['upload_directory']
    detected_path = json_config['default_parameters']['detected_directory']
    #checking if directories for uploaded and detected images exist if not create them
    check_dirs([upload_path, detected_path])

    #loading the model before start of server
    t0=time.perf_counter()
    detector = load_model(json_config['model_file']['default'], json_config['default_parameters']['default_speed']) 
    t1=time.perf_counter()

    print("Time elapsed in loading model:", t1-t0)

    app.run(debug=True, host= json_config['host'], port=json_config['port'])