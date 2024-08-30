import pickle
import numpy as np
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split, cross_val_score  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore


# --- GENERAL CONFIGURATION ---
data_dict = pickle.load(open('./data.pickle', 'rb'))     

#max_length = max(len(seq) for seq in data_dict['data'])
#data_padded = pad_sequences(data_dict['data'], maxlen=max_length, padding='post', truncating='post', dtype='float32')

# data = np.asarray(data_padded)
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


# --- TRAINING and TESTING SETS ---
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01, shuffle=True, stratify=labels)


# --- TRAINING AND TESTING THE CLASSIFIER ---
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classifed correctly '.format(score*100))


# --- SAVE THE DATA ---
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
