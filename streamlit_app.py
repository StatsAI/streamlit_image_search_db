# © Stats AI LLC 2023. All Rights Reserved. 
# No part of this code may be used or reproduced without express permission.

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
import time

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
	url = "https://github.com/StatsAI/streamlit_image_search_db/releases/download/image_search_assets/img_emb.pkl"
	with requests.get(url) as r:
		pickle_file = r.content

    	img_emb_loaded = pickle.loads(pickle_file)
    	return img_emb_loaded


# Load Pre-trained Assets
@st.cache_resource
def load_assets():
	# Load images from a folder
	image_list = load_data(['animals'])

	# Load indexed images
	img_emb_loaded = load_embeddings()

	# Load image features
	#image_data = pd.read_pickle("image_data_features.pkl")

	return image_list, img_emb_loaded


iimage_list, img_emb_loaded = load_assets()

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
st.write('This is a web app to demo reverse image search using the FAISS library.')
st.sidebar.write('Instructions: Use the below controls to select the Image you want to find similar images of') 

images_recs = st.sidebar.slider(label = 'Image Index', min_value = 0,
                          max_value = 5400,
                          value = 150,
                          step = 1)

image_path = image_list[images_recs]

with st.sidebar:
	# Display an image
        st.image(image_path)

####################################################################################################################################################

# Set up the search engine
@st.cache_resource
def load_model():
    model = Search_Setup(image_list=image_list,model_name='vgg19',pretrained=True,image_count= None)
    return model

s = load_model()

def _extract(img):
        # Resize and convert the image
        img = img.resize((224, 224))
        img = img.convert('RGB')

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224, 0.225]),
        ])
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

        # Extract features
        feature = s.model(x)
        feature = feature.data.numpy().flatten()
        return feature / np.linalg.norm(feature)


def _get_query_vector(image_path: str):
        img = Image.open(image_path)
        query_vector = _extract(img)
        return query_vector

def _search_by_vector(v, n: int):
        #self.v = v
        #self.n = n

        D, I = loaded_index.search(np.array([v], dtype=np.float32), n) #Image Path
        image_paths = [os.path.abspath(path) for path in image_data.iloc[I[0]]['images_paths'].to_list()] #Image Path
        #image_paths = [path.replace('/ImageSearch/drive/MyDrive/', '/') for path in image_paths] #Image Path
        #image_paths = [path.replace('drive/MyDrive/archive/animals/', '/') for path in image_paths] #Image Path

        return dict(zip(I[0], image_data.iloc[I[0]]['images_paths'].to_list()))
        #return image_paths

#@st.cache_resource
def plot_similar_images_new(image_path: str, number_of_images: int = 6):
        """
        Plots a given image and its most similar images according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image to be plotted.
        number_of_images : int, optional (default=6)
            The number of most similar images to the query image to be plotted.
        """
        input_img = Image.open(image_path)

        query_vector = _get_query_vector(image_path)
        img_list = list(_search_by_vector(query_vector, number_of_images).values())

        img_list = [path.replace('drive/MyDrive/archive/', '') for path in img_list] #Image Path

        img_list.append(image_path)

        img_list = list(set(img_list))

        number_of_images = 16
    
        grid_size = math.ceil(math.sqrt(number_of_images))
        axes = []
        fig = plt.figure(figsize=(20, 15))
        for a in range(number_of_images):
            axes.append(fig.add_subplot(grid_size, grid_size, a + 1))
            plt.axis('off')
            img = Image.open(img_list[a])
            img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
            plt.imshow(img_resized)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        #fig.suptitle('Similar Result Found', fontsize=22)


####################################################################################################################################################

if st.sidebar.button('Get Similar Images'):
	
	#candidate_predictions = retrieval_predict(num_recs, user_id)
	#st.session_state.results = st.pyplot(plot_similar_images_new(image_path = image_list[images_recs], number_of_images = 20))

	#fig, ax = plt.subplots()
	st.pyplot(plot_similar_images_new(image_path, number_of_images = 20))
	#plot_similar_images_new(image_path = image_list[images_recs], number_of_images = 20)
	

####################################################################################################################################################	





# Load Pre-trained Assets
@st.cache_resource
def load_assets():
	# Load images from a folder
	image_list = load_data(['animals'])

	# Load indexed images
	img_emb_loaded = load_embeddings()

	## Load image features
	#image_data = pd.read_pickle("image_data_features.pkl")

	return image_list, img_emb_loaded


image_list, loaded_index, image_data = load_assets()

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
st.write('This is a web app to demo reverse image search using the FAISS library.')
st.sidebar.write('Instructions: Use the below controls to select the Image you want to find similar images of') 

images_recs = st.sidebar.slider(label = 'Image Index', min_value = 0,
                          max_value = 5400,
                          value = 150,
                          step = 1)

image_path = image_list[images_recs]

with st.sidebar:
	# Display an image
        st.image(image_path)

####################################################################################################################################################

# Set up the search engine
@st.cache_resource
def load_model():
    model = Search_Setup(image_list=image_list,model_name='vgg19',pretrained=True,image_count= None)
    return model

s = load_model()

def _extract(img):
        # Resize and convert the image
        img = img.resize((224, 224))
        img = img.convert('RGB')

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224, 0.225]),
        ])
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

        # Extract features
        feature = s.model(x)
        feature = feature.data.numpy().flatten()
        return feature / np.linalg.norm(feature)


def _get_query_vector(image_path: str):
        img = Image.open(image_path)
        query_vector = _extract(img)
        return query_vector

def _search_by_vector(v, n: int):
        #self.v = v
        #self.n = n

        D, I = loaded_index.search(np.array([v], dtype=np.float32), n) #Image Path
        image_paths = [os.path.abspath(path) for path in image_data.iloc[I[0]]['images_paths'].to_list()] #Image Path
        #image_paths = [path.replace('/ImageSearch/drive/MyDrive/', '/') for path in image_paths] #Image Path
        #image_paths = [path.replace('drive/MyDrive/archive/animals/', '/') for path in image_paths] #Image Path

        return dict(zip(I[0], image_data.iloc[I[0]]['images_paths'].to_list()))
        #return image_paths

#@st.cache_resource
def plot_similar_images_new(image_path: str, number_of_images: int = 6):
        """
        Plots a given image and its most similar images according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image to be plotted.
        number_of_images : int, optional (default=6)
            The number of most similar images to the query image to be plotted.
        """
        input_img = Image.open(image_path)

        query_vector = _get_query_vector(image_path)
        img_list = list(_search_by_vector(query_vector, number_of_images).values())

        img_list = [path.replace('drive/MyDrive/archive/', '') for path in img_list] #Image Path

        img_list.append(image_path)

        img_list = list(set(img_list))

        number_of_images = 16
    
        grid_size = math.ceil(math.sqrt(number_of_images))
        axes = []
        fig = plt.figure(figsize=(20, 15))
        for a in range(number_of_images):
            axes.append(fig.add_subplot(grid_size, grid_size, a + 1))
            plt.axis('off')
            img = Image.open(img_list[a])
            img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
            plt.imshow(img_resized)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        #fig.suptitle('Similar Result Found', fontsize=22)


####################################################################################################################################################

if st.sidebar.button('Get Similar Images'):
	
	#candidate_predictions = retrieval_predict(num_recs, user_id)
	#st.session_state.results = st.pyplot(plot_similar_images_new(image_path = image_list[images_recs], number_of_images = 20))

	#fig, ax = plt.subplots()
	st.pyplot(plot_similar_images_new(image_path, number_of_images = 20))
	#plot_similar_images_new(image_path = image_list[images_recs], number_of_images = 20)
	

####################################################################################################################################################	


