# -*- coding: utf-8 -*-
"""Untitled.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LE_2FGnfuWSpO00B3lbvgyg8galV-Lge

# Import Libraries
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import re

"""# Loading The model"""

model = tf.keras.models.load_model('trained_model.keras')

model.summary()

"""# Visualizing the single image test set"""

!pip install opencv-python

import cv2
image_path = "test/Acne and Rosacea Photos/07sebDerem1101051.jpg"
#Reading Image
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Displaying Image
plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()

"""# Testing Model"""

image = tf.keras.preprocessing.image.load_img(image_path,target_size=(192, 192))
input_arr=tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
print(input_arr.shape)

prediction = model.predict(input_arr)
prediction,prediction.shape

result_index = np.argmax(prediction)
result_index

class_names = [ 'Acne and Rosacea Photos',
 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
 'Atopic Dermatitis Photos',
 'Bullous Disease Photos',
 'Cellulitis Impetigo and other Bacterial Infections',
 'Eczema Photos',
 'Exanthems and Drug Eruptions',
 'Herpes HPV and other STDs Photos',
 'Light Diseases and Disorders of Pigmentation',
 'Lupus and other Connective Tissue diseases',
 'Melanoma Skin Cancer Nevi and Moles',
 'Poison Ivy Photos and other Contact Dermatitis',
 'Psoriasis pictures Lichen Planus and related diseases',
 'Seborrheic Keratoses and other Benign Tumors',
 'Systemic Disease',
 'Tinea Ringworm Candidiasis and other Fungal Infections',
 'Urticaria Hives',
 'Vascular Tumors',
 'Vasculitis Photos',
 'Warts Molluscum and other Viral Infections']

# Extract the class name (directory name) and clean it
class_name = os.path.basename(os.path.dirname(image_path))
cleaned_class_name = class_name.replace(" Photos", "")  # Remove 'Photos' from the class name

# Extract the disease type from the filename (before the numbers and extension)
filename = os.path.basename(image_path)  # Get filename: '07sebDerem1101051.jpg'
type_name = re.findall(r'[a-zA-Z]+', filename.split('.')[0])[0]  # Extract 'sebDerem'

# Combine the disease name and type
full_disease_name = f"{cleaned_class_name} : {type_name}"

# Displaying Result of Disease Prediction
model_prediction = full_disease_name

# Plotting the image
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()

model_prediction

