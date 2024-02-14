import pickle
import numpy as np
from PIL import Image
from flask import Flask, render_template, request


app = Flask(__name__)
model_dict = pickle.load(open('model.p', 'rb'))
full_pipeline = model_dict['model']

skin_labels_dict = {    
    'actinic keratosis': 'Actinic Keratosis', 'basal cell carcinoma': 'Basal Cell Carcinoma', 
    'dermatofibroma': 'Dermatofibroma', 'melanoma': 'Melanoma', 'nevus': 'Nevus',
    'pigmented benign keratosis': 'Pigmented Benign Keratosis', 'seborrheic keratosis': 'Seborrheic Keratosis',
    'squamous cell carcinoma': 'Squamous Cell Carcinoma', 'vascular lesion': 'Vascular Lesion'
}
# IS CANCER: Basal Cell Carcinoma, Squamous Cell Carcinoma, Melanoma


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            img = img.resize((224, 224))
            img_array = np.array(img)

            prediction = full_pipeline.predict(img_array.reshape(1, -1))
            predicted_condition = skin_labels_dict.get(prediction[0], 'Unknown Condition')
            is_cancer = predicted_condition in ['Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Melanoma']

            return render_template('result.html', predicted_condition=predicted_condition, is_cancer=is_cancer)
        
    return 'Error'


if __name__ == '__main__':
    app.run(debug=True)
