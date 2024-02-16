# Try the app! https://appimagesearchdb-bghvgku4fhakucrz2xeeum.streamlit.app/

# How it works: 
## Step 0: Embeddings of 5,000 animals are created using Google colab and uploaded to Qdrant Cloud Vector Database. 
## Step 1: The user selects an animal image or enters its name in the textbox. 
## Step 2: The user enters an OpenAI API Key or leaves it blank. 
## Step 3: The image or text is then transformed into a query embedding. 
## Step 4: The query embedding is used to perform a nearest neighbors similiarty search in Qdrant.
## Step 5: The animal type is determined via the payload of the first returned result from Qdrant.  
## Step 6: The images returned are pulled via Qdrant payload (image link) + Github (image storage).
## Step 7: If the user entered an OpenAI API Key, then the output of Step 5 is passed to OpenAI for a animal summary. 

# Feature 1: Text to Image Search!

![image](https://github.com/StatsAI/streamlit_image_search_db/assets/67183539/190b6e85-90be-464b-9d7f-d6f86de73d73)
![image](https://github.com/StatsAI/streamlit_image_search_db/assets/67183539/2a742aaf-5f95-4dd9-b6ba-3ce7db61a5c1)

# Feature 2: Image to Image Search!

![image](https://github.com/StatsAI/streamlit_image_search_db/assets/67183539/876f3bd7-0051-489b-8141-98f3565c069c)
![image](https://github.com/StatsAI/streamlit_image_search_db/assets/67183539/ed57ca40-8474-4118-8b45-716f3d46f02a)











