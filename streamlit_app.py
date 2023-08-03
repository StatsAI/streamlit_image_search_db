
import pandas as pd
import streamlit as st
import numpy as np 
import tensorflow as tf
from PIL import Image


# load tensorflow models for retrieval and ranking
loaded_retrieval_model = tf.saved_model.load('models/index_model')
loaded_ranking_model = tf.saved_model.load('models/ranking_model')

# this works
#model1 = tf.keras.models.load_model('models/my_keras_model1.h5')

####################################################################################################################################################

st.title('Tensorflow Recommenders Library Movie Recommendation System')

st.write('This is a web app to recommend movies to users based upon their watch history using the Tensorflow Recommenders Python library.')

st.write('It uses the following two-model approach, as outlined by: [Tensorflow Recommenders](https://www.tensorflow.org/recommenders/examples/basic_retrieval)')         

#opening the image

image = Image.open('images/rec_sys.PNG')

#displaying the image on streamlit app

st.image(image)

st.sidebar.write('Instructions: Use the below controls to select the number of movie recommendations you would like to generate for a given user.')

user_id = st.sidebar.selectbox(label = 'Select the user ID', options = ('1', '2', '3', '4','5','6','7','8','9','10'), index= 0)

num_recs = st.sidebar.slider(label = 'Number of Recommendations', min_value = 1,
                          max_value = 5 ,
                          value = 3,
                          step = 1)

####################################################################################################################################################

def retrieval_predict(num_recs, user_id):
	
	scores, titles = loaded_retrieval_model([user_id])
		
	titles = titles.numpy()[0]
	
	holder = []
	
	for title in titles:
	
		title = str(title)
		title = title.replace('b', '')
		title = title.strip()
		#title = title.replace('"', "")
		title = title.replace("'","")
		holder.append(title)		
	
	#return holder
	#return holder[:2]
	return holder[:num_recs]

	
	
def ranking_predict(user_id, candidate_predictions):
	result = {}
	
	test_movie_titles = candidate_predictions 
	
	#for title in test_movie_titles:
	#	title = title.replace('"', '')
	
	#test_movie_titles = ['Deep Rising (1998)', 'Sphere (1998)','Fallen (1998)','Hard Rain (1998)','Jackie Brown (1997)']
		
	#test_movie_titles = ['Grand Day Out, A (1992)', 'Blue in the Face (1995)', 'Hudsucker Proxy, The (1994)', 'Crum (1994)', 'Close Shave, A (1995)']
	
	for movie_title in test_movie_titles:
		result[movie_title] = loaded_ranking_model({
		"user_id": np.array([user_id]),
		"movie_title": np.array([movie_title])
		})
		
	#result = pd.DataFrame.from_dict(result)
	
	#holder = {}

	#for title, score in sorted(result.items(), key=lambda x: x[1], reverse=True):	
	#	holder[movie_title] = [title, score]

	#result = result.numpy()[0]
	
	#return type(list(result.values())[0])
	
	#score.numpy()[0][0]
	
	#return result.values()

	#return list(result.values())[0].numpy()[0][0]

	#return list(result)[:num_recs]
	
	#return result.values()
	
	return result
	

####################################################################################################################################################

# Initialize session state

#if "load_state" not in st.session_state:
#	st.session_state.load_state = False

if "candidate_predictions" not in st.session_state:
	st.session_state.candidate_predictions = None

st.sidebar.write('Instructions: Click on the generate candidates button to generate a list of candidates using the retrieval model.')

#st.sidebar.button('Generate Candidates', key = "1")

if st.sidebar.button('Generate Candidates'):
	
	#candidate_predictions = retrieval_predict(num_recs, user_id)
	st.session_state.candidate_predictions = retrieval_predict(num_recs, user_id)
	st.write('Your candidate recommendations are: ' + str(st.session_state.candidate_predictions))
	

####################################################################################################################################################	

st.sidebar.write('Instructions: Click on the rank candidates button to rank the candidates using the ranking model.')

if st.sidebar.button('Rank Candidates'):
	
	ranking_predictions = ranking_predict(user_id, st.session_state.candidate_predictions)
	
	st.write('Your candidate recommendations are: ' + str(st.session_state.candidate_predictions))
	st.write('Your candidate rankings are: ' + str(ranking_predictions))
