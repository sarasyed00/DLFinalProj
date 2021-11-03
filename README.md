# Predicting Yelp Reviews Based on Sentiment of Text

**How to Access dataset:**
In order to access the yelp reviews dataset we used, download the set from this google drive link:

https://drive.google.com/drive/folders/1MX-j9iuzJkvFx3K8Wg-IcHOpfy454Eg9?usp=sharing

**Referencing File Path:**
To reference the correct file path, change the first line of the main function to the correct file path as the argument to the get_data() function.  

**Structure:**
Our project relied on two main files: preprocessing.py and assignment.py.

Within preprocessing we use the get_data() function to read the file path, clean some of the text and choose how many reviews we want to extract from the dataset. We then use tokenize_with_keras() to tokenize the words and embedding_matrix() to utilize a Keras embedding embedding matrix.

Within assignment.py is our main model architecture. We create the Recurrent neural network that relies on lstm and dense layers. We train and test within assignment.py. Our program prints out the accuracy of the model at the end of its execution.
