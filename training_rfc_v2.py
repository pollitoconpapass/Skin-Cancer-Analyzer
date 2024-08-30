import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# === LOAD THE DATA ===
data_dict = pickle.load(open('./data.pickle', 'rb')) 

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


# === TRAINING AND TESTING SETS ===
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01, shuffle=True, stratify=labels)


# === INITIALIZE INDIVIDUAL CLASSIFIERS ===
rf_classifier = RandomForestClassifier()
gb_classifier = GradientBoostingClassifier()

rf_classifier.fit(x_train, y_train)
gb_classifier.fit(x_train, y_train)

rf_pred = rf_classifier.predict(x_test)
gb_pred = gb_classifier.predict(x_test)


# === INDIVIDUAL ACCURACY ===
rf_accuracy = accuracy_score(y_test, rf_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)

print("Random Forest Classifier Accuracy:", rf_accuracy)
print("Gradient Boosting Classifier Accuracy:", gb_accuracy)


# === ENSEMBLE USING VotingClassifier ===
ensemble_classifier = VotingClassifier(estimators=[
    ('random_forest', rf_classifier),
    ('gradient_boosting', gb_classifier)
], voting='hard')  # 'hard' voting uses majority voting

ensemble_classifier.fit(x_train, y_train)
ensemble_pred = ensemble_classifier.predict(x_test)

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print("\nEnsemble Classifier Accuracy:", ensemble_accuracy)


# === SAVE THE MODEL ===
with open('ensemble_model.p', 'wb') as f:
    pickle.dump({'model': ensemble_classifier}, f)
