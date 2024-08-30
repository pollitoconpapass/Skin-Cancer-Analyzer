import pickle
import numpy as np
import streamlit as st
from PIL import Image


model_dict = pickle.load(open('model.p', 'rb'))
full_pipeline = model_dict['model']

skin_labels_dict = {    
    'actinic keratosis': 'Actinic Keratosis', 'basal cell carcinoma': 'Basal Cell Carcinoma', 
    'dermatofibroma': 'Dermatofibroma', 'melanoma': 'Melanoma', 'nevus': 'Nevus',
    'pigmented benign keratosis': 'Pigmented Benign Keratosis', 'seborrheic keratosis': 'Seborrheic Keratosis',
    'squamous cell carcinoma': 'Squamous Cell Carcinoma', 'vascular lesion': 'Vascular Lesion'
}
# IS CANCER: Basal Cell Carcinoma, Squamous Cell Carcinoma, Melanoma

st.title('Skin Condition Classifier')
file = st.file_uploader('Upload Image', type=['jpg', 'png'])

if file:
    img = Image.open(file)
    img = img.resize((224, 224))
    img_array = np.array(img)

    prediction = full_pipeline.predict(img_array.reshape(1, -1))
    predicted_condition = skin_labels_dict.get(prediction[0], 'Unknown Condition')
    is_cancer = predicted_condition in ['Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Melanoma']

    st.image(img, use_column_width=True)
    st.write("\nPREDICTED CONDITION:")
    if is_cancer:
        st.warning(f'{predicted_condition} - CANCER DETECTED')
    else:
        st.success(f'{predicted_condition} - NO CANCER DETECTED')
