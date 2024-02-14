# Skin Condition Classifier
### Steps Guide
1. Download the Skin Cancer ISIC dataset: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
2. Copy the folder images inside a folder named ´´´data´´´ inside the root
   (choose one of the folders or training or testing)
3. Run the create set file: ´´´python create_set.py´´´
4. Run the training file: ´´´python training.py´´´ (currently in progress)
5. Configure the Flask application:
   - ´´´export FLASK_APP=app.py´´´
   - ´´´export FLASK_DEBUG=1´´´
   - ´´´flask run´´´
