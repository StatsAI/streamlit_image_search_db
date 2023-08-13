# © Stats AI LLC 2023. All Rights Reserved. 
# No part of this code may be used or reproduced without express permission.

import streamlit as st
#from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torch
#from torch.autograd import Variable
import os
import math
import time
import uuid
import requests
import zipfile
import json

import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer, util

#from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
import pickle


st.set_option('deprecation.showPyplotGlobalUse', False)

####################################################################################################################################################

# Download and unzip images
@st.cache_resource
def download_and_unzip(url):
	response = requests.get(url)
	with open("archive.zip", "wb") as f:
        	f.write(response.content)
	
	with zipfile.ZipFile("archive.zip", "r") as zip_ref:
        	zip_ref.extractall()

def load_data(folder_list: list):
	image_path = []
	
	for folder in folder_list:
		for root, dirs, files in os.walk(folder):
			for file in files:
				if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
					image_path.append(os.path.join(root, file))
	return image_path

def load_embeddings():
	url = "https://github.com/StatsAI/streamlit_image_search_db/releases/download/image_search_assets/img_dict.txt"

	# Download the file
	response = requests.get(url)

	# Load the file into a string
	file_content = response.content.decode("utf-8")

	# Create a dictionary from the string
	img_dict = json.loads(file_content)
	
	img_list = list(img_dict.keys())
	img_emb = list(img_dict.values())
		
	return img_list, img_emb


## Load Pre-trained Assets
@st.cache_resource
def load_assets():
	# Load images from a folder
	#image_list = load_data(['animals'])

	# Load indexed images
	image_list, img_emb_loaded = load_embeddings()
	img_emb_loaded = torch.tensor(img_emb_loaded)

	return image_list, img_emb_loaded

# Set up the search engine
@st.cache_resource
def load_model():
	model = SentenceTransformer("clip-ViT-B-32")
	return model
	
####################################################################################################################################################

url = "https://github.com/StatsAI/streamlit_image_search_db/releases/download/image_search_assets/archive.zip"
download_and_unzip(url)

image_list, img_emb_loaded = load_assets()

model = load_model()

#st.write(type(img_emb_loaded))

####################################################################################################################################################

logo = Image.open('images/picture.png')
#newsize = (95, 95)
#logo = logo.resize(newsize)

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
	    padding-top: 0;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.image(logo)


st.markdown("""
        <style>
               .block-container {
		    padding-top: 0;
                }
        </style>
        """, unsafe_allow_html=True)

st.write('')
st.write('')
st.title('Image Recommendation App')
st.write("This app performs reverse image search using OpenAI's CLIP + Qdrant Vector Database")
st.sidebar.write('Use either option to find similar images!') 

images_recs = st.sidebar.slider(label = 'Image Search: Select an animal using the slider', min_value = 1,
                          max_value = 5400,
                          value = 1859,
                          step = 1)

image_path = image_list[images_recs - 1]

with st.sidebar:
	# Display an image
        st.image(image_path)

text_input = st.sidebar.text_input("Text Search: Enter animal. (Delete input to use slider)", "", key = "text")

a = ''
####################################################################################################################################################

#@st.cache_resource
def plot_similar_images_new(image_path, text_input, number_of_images: int = 6):
	
	animal_embedding = model.encode(image_path)	

	if text_input:
		animal_embedding = model.encode(text_input)
	
	animal_embedding = torch.tensor(animal_embedding)

	# Find the top 10 most similar images to the bear embedding.
	most_similar_images = util.semantic_search(query_embeddings = animal_embedding, corpus_embeddings = img_emb_loaded, top_k = number_of_images)

	# Create a list to store the results.
	results = []

	# Loop over the images in the most_similar_images variable.
	for i in range(len(most_similar_images[0])):
		# Get the image ID and score of the current image.
  		image_id = most_similar_images[0][i]['corpus_id']
  		image_score = most_similar_images[0][i]['score']

  		results.append([image_id, image_score])
	
	grid_size = math.ceil(math.sqrt(number_of_images))
	axes = []
	fig = plt.figure(figsize=(20, 15))
        
	for i in range(len(results)):
  		axes.append(fig.add_subplot(grid_size, grid_size, i + 1))
  		plt.axis('off')
  		image_number = results[i][0]
  		image_name = image_list[image_number]
  		score = results[i][1]
  		img = Image.open(image_name)
  		img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
  		plt.imshow(img_resized)
	#plt.title(f"Image {i}: {score}", fontsize=18)
	fig.tight_layout()
	fig.subplots_adjust(top=0.93)


####################################################################################################################################################

if st.sidebar.button('Get Similar Images'):
	st.pyplot(plot_similar_images_new(image_path, text_input, number_of_images = 16))
	#text_input = ""
	#text_input = st.sidebar.text_input("Input window", "", key = "text", on_change=clear_input_box)	

# if st.sidebar.button('Get Similar Images'):
	
# 	left_col, right_col = st.beta_columns(2)

# 	with left_col:
# 		text_input = st.sidebar.text_input("Input window", "", key = "text", on_change=clear_input_box)

# 	with right_col:
# 		st.button("Search")

####################################################################################################################################################	
