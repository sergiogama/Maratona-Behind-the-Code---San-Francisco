from watson_machine_learning_client import WatsonMachineLearningAPIClient
import wiotp.sdk.application
from flask import Flask, render_template, request, json, jsonify
import os
import json
import numpy as np
import io
from PIL import Image
import base64

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
    iotData = json.loads(base64.b64decode(lastEvent['payload']).decode('utf-8'))['data']

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
    ai_parms = { "wml_credentials" : wml_credentials, "model_endpoint_url" : "https://us-south.ml.cloud.ibm.com/v3/wml_instances/a8f286bb-6e41-4ddf-a97b-340ffcd6d33e/deployments/2193bbe9-3f93-4330-8b5a-11c202b1d795/online" }
    model_payload = { "values" : image }
    model_result = client.deployments.score( ai_parms["model_endpoint_url"], model_payload )
    data = model_result
    classes = ['CLEAN', 'DIRTY']
    index = data['values'][0][0].index(max(data['values'][0][0]))
    
    resposta = {
        "class": classes[index]
    }
    return resposta

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)