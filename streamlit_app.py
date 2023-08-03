import streamlit as st
import DeepImageSearch.config as config
from DeepImageSearch import Load_Data, Search_Setup
import requests
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torch
from torch.autograd import Variable
import timm
from PIL import ImageOps
import math
import faiss


####################################################################################################################################################
# Download and unzip images

def download_and_unzip(url):
    response = requests.get(url)
    with open("archive.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("archive.zip", "r") as zip_ref:
        zip_ref.extractall()

if __name__ == "__main__":
    url = "https://github.com/StatsAI/streamlit_image_search/releases/download/image_search_assets/archive.zip"
    download_and_unzip(url)

####################################################################################################################################################

# Load images from a folder
image_list = Load_Data().from_folder(['animals'])

# Load indexed images
loaded_index = faiss.read_index("https://github.com/StatsAI/streamlit_image_search/releases/download/image_search_assets/image_features_vectors.idx")

# Load image features
image_data = pd.read_pickle("https://github.com/StatsAI/streamlit_image_search/releases/download/image_search_assets/image_data_features.pkl")

####################################################################################################################################################

st.title('Image Recommendation App')

st.write('This is a web app to demo reverse image search using the FAISS library.')

#s.write('It uses the following two-model approach, as outlined by: [Tensorflow Recommenders](https://www.tensorflow.org/recommenders/examples/basic_retrieval)')         

#opening the image

#image = Image.open('images/rec_sys.PNG')

#displaying the image on streamlit app

#st.image(image)

st.sidebar.write('Instructions: Use the below controls to select the Image you want to find similar images of') 

images_recs = st.sidebar.slider(label = 'Image Index', min_value = 0,
                          max_value = len(image_list) ,
                          value = 150,
                          step = 1)

####################################################################################################################################################

# Set up the search engine
s = Search_Setup(image_list=image_list,model_name='vgg19',pretrained=True,image_count= None)

