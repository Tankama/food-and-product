import streamlit as st
from PIL import Image
import requests
import random
import pickle
import numpy as np
import io
from google_images_search import GoogleImagesSearch
import warnings
import os
from tensorflow.keras.models import model_from_json

warnings.filterwarnings('ignore')

# Initialize session state variables
if 'model' not in st.session_state:
    try:
        # Load model architecture
        with open(r'C:\Users\Tanusree\Desktop\Course_website\Ecommerce-product-image-classification-master\model_architecture_.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        
        # Create and load model weights
        st.session_state.model = model_from_json(loaded_model_json)
        st.session_state.model.load_weights(r"C:\Users\Tanusree\Desktop\Course_website\Ecommerce-product-image-classification-master\model_weights_.h5")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if 'classes_list' not in st.session_state:
    try:
        with open('class_names.pkl', 'rb') as pred_file:
            st.session_state.classes_list = pickle.load(pred_file)
    except Exception as e:
        st.error(f"Failed to load class names: {str(e)}")
        st.stop()

# Initialize global image list
if 'global_image_list' not in st.session_state:
    st.session_state.global_image_list = []

def get_file_content_as_string(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Failed to read file: {str(e)}")
        return ""

# UI Layout
st.title('Ecommerce product image classification for CDiscount.com')
Main_image = st.image('https://i.imgur.com/3UtpvAy.png', caption='Source: https://www.kaggle.com/c/cdiscount-image-classification-challenge/overview')

# Load instructions
try:
    instructions_path = os.path.join(os.path.dirname(__file__), 'Instructions.md')
    readme_text = st.markdown(get_file_content_as_string(instructions_path), unsafe_allow_html=True)
except:
    readme_text = st.markdown("Instructions not found", unsafe_allow_html=True)

# Sidebar menu
st.sidebar.markdown('# M E N U')
option = st.sidebar.selectbox('Choose the app mode', ('Show instructions', 'Run the app', 'Source code'))

def about():
    st.sidebar.markdown("# A B O U T")
    st.sidebar.image("https://i.imgur.com/dKOH8ry.png", width=180)
    st.sidebar.markdown("## Rohan Vailala Thoma")
    st.sidebar.markdown('* ####  Connect via [LinkedIn](https://in.linkedin.com/in/rohan-vailala-thoma)')
    st.sidebar.markdown('* ####  Connect via [Github](https://github.com/Rohan-Thoma)')
    st.sidebar.markdown('* ####  rohanvailalathoma@gmail.com')

def predict():
    st.write("-" * 34)
    number_of_images = len(st.session_state.global_image_list)
    st.write('#### Total products categorized:', number_of_images)
    st.write('-' * 34)
    
    if len(st.session_state.predictions) < number_of_images:
        try:
            pred_image = st.session_state.global_image_list[-1]
            pred_image = pred_image.resize((128, 128))
            pred_image = np.expand_dims(np.array(pred_image), axis=0) / 255.0
            
            pred = st.session_state.model.predict(pred_image, verbose=0)
            
            pred_list = []
            sorted_indices = np.argsort(pred[0])
            for h in range(4):
                pred_index = sorted_indices[-1 - h]
                predicted_label = st.session_state.classes_list[pred_index]
                probability = np.round(pred[0][pred_index] * 100, 3)
                pred_list.append([predicted_label, probability])
            st.session_state.predictions.append(pred_list)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return
    
    for i in range(number_of_images):
        try:
            image_ = st.session_state.global_image_list[-1 - i]
            preds_ = st.session_state.predictions[-1 - i]
        except:
            continue
        
        col1, col2, col3 = st.columns([1.5, 2, 1])
        
        with col1:
            if i == 0:
                st.write("### Image")
                st.write("-" * 40)
            st.image(image_, width=180)

        with col2:
            if i == 0:
                st.write("### Product Category")
                st.write("-" * 40)
            for g in range(4):
                st.write('* ', preds_[g][0].upper())
        
        with col3:
            if i == 0:
                st.write("### Confidence")
                st.write("-" * 40)
            for g in range(4):
                st.write('* ', preds_[g][1], ' %')
        
        st.write('-' * 34)

# App modes
if option == 'Show instructions':
    st.sidebar.success('To continue, select "Run the app"')
    st.sidebar.warning('To see the code, go to "Source code"')
    about()

elif option == 'Source code':
    Main_image.empty()
    readme_text.empty()
    st.sidebar.success('To continue, select "Run the app"')
    st.sidebar.warning('Go to "Show instructions" to read more about the app')
    
    try:
        code_path = os.path.join(os.path.dirname(__file__), 'app_code.txt')
        with open(code_path, 'r', encoding='utf-8') as text_file:
            st.code(text_file.read())
    except Exception as e:
        st.error(f"Failed to load source code: {str(e)}")
    
    about()

elif option == 'Run the app':
    Main_image.empty()
    readme_text.empty()
    st.sidebar.warning('Go to "Show instructions" to read more about the app')
    st.sidebar.success('To see the code, go to "Source code"')
    about()
    
    st.markdown('### Choose your preferred method')
    genre = st.radio("", ('Upload a product image yourself', 'Get a random product image from google automatically'))
    st.write("-" * 34)
    
    if genre == 'Upload a product image yourself':
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                image = image.resize((512, 512))
                st.session_state.global_image_list.append(image)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(image, width=250)
                with col2:
                    st.success('Successfully uploaded the image. Please see the category predictions below')
                    st.write("-" * 34)
                    st.warning('To get a random image from the web, press the 2nd option above..!')
                
                with st.spinner("Getting predictions..."):
                    predict()
            except Exception as e:
                st.error('Please upload a valid image file (jpg, jpeg, or png)')
    
    elif genre == 'Get a random product image from google automatically':
        with st.spinner("Getting a random product image from google..."):
            try:
                with open('search_list.pkl', 'rb') as f:
                    search_list = pickle.load(f)
                
                search_word = 'buy ' + random.choice(search_list) + ' items online amazon'
                _search_params = {
                    'q': search_word,
                    'num': 10,
                    'fileType': 'jpg|gif|png'
                }
                
                # Use only one API key (replace with your actual key)
                gis = GoogleImagesSearch('YOUR_API_KEY', 'YOUR_CX')
                gis.search(search_params=_search_params)
                
                hdr = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if gis.results():
                        im = random.choice(gis.results())
                        try:
                            image_file = requests.get(im.url, headers=hdr, timeout=10).content
                            image = Image.open(io.BytesIO(image_file)).convert('RGB')
                            image = image.resize((512, 512))
                            st.session_state.global_image_list.append(image)
                            st.image(image, width=250)
                        except:
                            st.error("Could not load the image from URL")
                with col2:
                    if st.button('Try with another image'):
                        st.experimental_rerun()
                    st.success("Got a random product image from the web. Please see the predictions below..!")
                    st.warning("Choose 'upload an image' to input your custom image..!")
                
                with st.spinner("Getting predictions..."):
                    predict()
            except Exception as e:
                st.error(f"Failed to get image from Google: {str(e)}")
         
       

    

