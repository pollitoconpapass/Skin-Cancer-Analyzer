import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator   # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore


# --- GENERAL CONFIGURATION ---
data_dict = pickle.load(open('./data.pickle', 'rb'))
# print(data_dict)

# Assuming the images are already preprocessed and resized to a fixed size
images = np.array(data_dict['data'])
label_mapping = {'melanoma': 0, 'pigmented benign keratosis': 1, 'nevus': 2, 
                 'basal cell carcinoma': 3, 'actinic keratosis': 4, 
                 'squamous cell carcinoma': 5, 'vascular lesion': 6, 
                 'seborrheic keratosis':7, 'dermatofibroma': 8}

labels = np.array([label_mapping[label] for label in data_dict['labels']])

image_height = 224
image_width = 224
num_channels = 3  # Assuming RGB images
images = images.reshape(-1, image_height, image_width, num_channels)


# --- TRAINING and TESTING SETS ---
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=True, stratify=labels)


# --- DATA AUGMENTATION ---
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255
)

train_generator = train_datagen.flow(x_train, y_train, batch_size=32)


# --- CNN MODEL ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# --- TRAINING THE CNN ---
history = model.fit(train_generator, epochs=20, validation_data=(x_test, y_test))


# --- EVALUATE THE MODEL ---
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test Accuracy:', test_accuracy*100)


# --- SAVE THE MODEL ---
model.save('skin_cancer_cnn_model.h5')
