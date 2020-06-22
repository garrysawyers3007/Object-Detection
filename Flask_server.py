from Functions import *

app = Flask(__name__)

detector = None 
json_config = None
logger(app)

@app.route('/image/', methods=['POST'])
def image_request():
    if(request.method=='POST'):

        json_body= request.get_json()

        image_name, image_type, image_file, objects, minimum_probability, detection_speed = handle_request(json_body, json_config)
        
        image_path = upload_image(json_config, image_file, image_name, image_type)

        t0=time.perf_counter()
        detections = Detection(json_config, detector, objects, image_path, minimum_probability)
        t1=time.perf_counter()
        
        print("Time elapsed in detection:", t1-t0)

        return jsonify(
            objects = json.dumps(detections, default= int)
        )
         
@app.route('/', methods=['GET'])
def hello():
    return "Hello"

if __name__ == '__main__':

    json_config = get_config('config.json')

    #logger(json_config, app)
    upload_path = json_config['default_parameters']['upload_directory']
    detected_path = json_config['default_parameters']['detected_directory']

    check_dirs([upload_path, detected_path])

    t0=time.perf_counter()
    detector = load_model(json_config['model_file']['default'], json_config['default_parameters']['default_speed']) 
    t1=time.perf_counter()

    print("Time elapsed in loading model:", t1-t0)

    app.run(debug=True, host= json_config['host'], port=json_config['port'])