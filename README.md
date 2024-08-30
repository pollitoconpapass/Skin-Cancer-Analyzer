# Skin Condition Classifier - Streamlit frontend

### Steps to follow
1. Download the Skin Cancer ISIC dataset from [here](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

2. Inside the root project, create a folder named `data` and copy the folder images inside it

3. Install all the requirements: 

        pip install -r requirements.txt

4. Create the set file for the training:

        python create_set.py


5. Start the training

        python training.py

    There are other training files for you to choose `training_cnn.py` and `training_rfc_v2.py`. Choose the most appeal one.

6. Start the frontend

        streamlit run app.py
