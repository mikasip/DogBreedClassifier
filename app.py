import label_image_ws as cl
from flask import Flask, request, render_template, jsonify, url_for
from flask_restful import Api, Resource
import json
import os

app = Flask(__name__, static_url_path='/static')
api = Api(app)
classifier = cl.Classifier()

@app.route('/')
def index():
    return render_template('index.html')

app.config['UPLOAD_FOLDER'] = "static/uploads"

@app.route('/upload', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        try:
            f = request.files['image']
            filename = f.filename
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return os.path.join(app.config['UPLOAD_FOLDER'], filename), 200
        except Exception as err:
            return err
      
@app.route('/delete', methods = ['POST'])
def delete_image():
    if request.method == 'POST':
        try:
            data = request.get_json(force = True)
            file_path = data['imagePath']
            os.remove(file_path)
            return "success", 200
        except:
            return "error", 500

class Classification(Resource):
    def post(self):
        try:
            data = request.get_json(force = True)
        except:
            return {'errorMessage': 'Wrong request...'}, 500
        if(data != None and len(data)>0):
            response, err = classifier.classify(data['imageURL'], data['local'])
        else:
            return {'errorMessage': 'Please, provide a fileName...'}, 500
        if isinstance(response, list):
            return json.dumps(response), 200
        else:
            return {'errorMessage': err}, 404

api.add_resource(Classification, '/classify')

if __name__ == "main":
    app.run()