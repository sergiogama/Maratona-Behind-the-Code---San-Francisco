from watson_machine_learning_client import WatsonMachineLearningAPIClient
import wiotp.sdk.application
from flask import Flask, render_template, request, json, jsonify
import os
import json
import numpy as np
import io
from PIL import Image
import base64

wml_credentials={
  "apikey": "F4-lBflj4TwuBT8Yj21qiCIN1X7UPTMIYN2mDXa-FuxW",
  "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance - crn:v1:bluemix:public:pm-20:us-south:a/5086f0d78cb04b3aabd7b046c5c84a10:a8f286bb-6e41-4ddf-a97b-340ffcd6d33e::",
  "iam_apikey_name": "auto-generated-apikey-e56daaf9-fb14-47ae-890a-b700d8a03a7d",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/5086f0d78cb04b3aabd7b046c5c84a10::serviceid:ServiceId-c2399bed-c92e-4d5f-9c35-d5aae1725952",
  "instance_id": "a8f286bb-6e41-4ddf-a97b-340ffcd6d33e",
  "url": "https://us-south.ml.cloud.ibm.com"
}

app = Flask(__name__)
app.config.from_object(__name__)
port = int(os.getenv('PORT', 8080))

@app.route("/", methods=['GET'])
def hello():
    error=None
    return render_template('index.html', error=error)

@app.route("/iot", methods=['GET'])
def result():
    print(request)
    
    # Receber dados IoT
    device = {"typeId": "maratona", "deviceId": "maratona_1"}
    eventId = 'sensor'
    options = { "auth": {
        "key": "a-dhpz8n-9qnqykz0na",
        "token": "CN0vxYYN7?8AZmsJRr"
    }}
    appClient = wiotp.sdk.application.ApplicationClient(options)
    lastEvent = appClient.lec.get(device, eventId)
    print(lastEvent)
    iotData = json.loads(base64.b64decode(lastEvent['payload']).decode('utf-8'))

    # Calculos
    # ITU = T - 0.55 (1 - UR )( T - 14 )
    itu = iotData['temperatura'] - (0.55 * (1 - iotData['umidade_ar']) * (iotData['temperatura'] - 14))
    
    # Volume Agua = (4 * pi * r^3)/2 * us
    va = ((4 * 3.14)/6) * iotData['umidade_solo']
    
    # Fahrenheit f = (c * 9/5) + 32
    fr = (iotData['temperatura']*(9/5)) + 32
    
    resposta = {
        "iotData": iotData,
        "itu": itu,
        "volumeAgua": va,
        "fahrenheit": fr
    }
    response = app.response_class(
        response=json.dumps(resposta),
        status=200,
        mimetype='application/json'
    )
    return response

def prepare_image(image):
    image = image.resize(size=(96,96))
    image = np.array(image, dtype="float") / 255.0
    image = np.expand_dims(image,axis=0)
    image = image.tolist()
    return image

@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    image = request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image)

    # Faça uma requisição para o serviço Watson Machine Learning aqui e retorne a classe detectada na variável 'resposta'
    client = WatsonMachineLearningAPIClient( wml_credentials )
    ai_parms = { "wml_credentials" : wml_credentials, "model_endpoint_url" : "https://us-south.ml.cloud.ibm.com/v3/wml_instances/a8f286bb-6e41-4ddf-a97b-340ffcd6d33e/deployments/610327f1-d185-4094-ac38-98d5b6bfba0b/online" }
    model_payload = { "values" : image }
    model_result = client.deployments.score( ai_parms["model_endpoint_url"], model_payload )
    data = model_result
    classes = ['CLEAN', 'DIRTY']
    print(data)
    index = data['values'][0][0].index(max(data['values'][0][0]))
    
    resposta = {
        "class": classes[index]
    }
    return resposta

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)