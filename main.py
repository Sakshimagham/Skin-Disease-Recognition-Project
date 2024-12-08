import streamlit as st
import tensorflow as tf
import numpy as np
import os
import re

# TensorFlow Model Prediction
# TensorFlow Model Prediction
def model_prediction(test_image):
    # Load the model
    model = tf.keras.models.load_model('trained_model.keras')
    
    # Convert the uploaded image to the correct format
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(192, 192))
    
    # Convert image to array and expand the dimensions for prediction
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    
    # Make prediction
    prediction = model.predict(input_arr)

    # Get the predicted class index
    result_index = np.argmax(prediction)
    
    return result_index

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# Home Page
if(app_mode == "Home"):
    st.header("SKIN DISEASE RECOGNITION SYSTEM")
    image_path = "web-page.image.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Skin Disease Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying skin diseases efficiently. Upload an image of your injured part, and our system will analyze it to detect any signs of diseases. Together, let's make "Healing Skin,Healing Lives".
    ### How It Works
    1. *Upload Image:* Go to the *Disease Recognition* page and upload an image of your injured part with suspected diseases.
    2. *Analysis:* Our system will process the image using advanced algorithms to identify potential diseases.
    3. *Results:* View the results and recommendations for further action.

    ### Why Choose Us?
    - *Accuracy:* Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - *User-Friendly:* Simple and intuitive interface for seamless user experience.
    - *Fast and Efficient:* Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the *About* page.

    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset consists of images representing 20 different skin diseases, sourced from Kaggle. Unlike the original dataset with predefined training, validation, and test sets, this dataset has been split into training and validation subsets directly from the available data. The split ensures an appropriate distribution of data for effective model training and evaluation. Data augmentation techniques were applied offline to increase the diversity of the training set, helping the model generalize better to unseen data.
                
                """)
    


#Prediction Page

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        # Display the uploaded image
        st.image(test_image, use_column_width=True)
        
        # When Predict button is clicked
        if st.button("Predict"):
            with st.spinner("Please Wait..."):
                # Get model prediction
                result_index = model_prediction(test_image)

            # List of class names corresponding to the output indices
            class_names = [
                'Acne and Rosacea Photos',
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
                'Warts Molluscum and other Viral Infections'
            ]

            # Display result after the prediction
            st.success(f"The model predicts that the image is a type of: {class_names[result_index]}")

            # Trigger animations after prediction
            st.balloons()
            st.snow()  # Optional: Show snow animation for effect