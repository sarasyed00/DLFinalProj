import json
import string
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def get_data(file_path):
    reviews = []
    ratings = []
    items = []
    x = 0
    print("Started Reading JSON file which contains multiple JSON items")
    with open(file_path, encoding="utf8") as f: #loading 100 reviews
        counter = 0
        for jsonObj in f:
            counter += 1
            item = json.loads(jsonObj)
            items.append(item)
            if counter == 500:
                break
    for item in items:
        text = item["text"]
        text = clean_review(text) #cleans the review
        reviews.append(text)
        stars = item["stars"]
        ratings.append(stars)
    ratings = np.array(list(map(lambda x: 1 if x == 3.0 or x == 4.0 or x == 5.0 else 0, ratings))) #binary ratings
    return reviews, ratings

def clean_review(review): #method that cleans a review by getting rid of punctuation and splitting on each word
    text = review.split()  # tokenizing the text, split by words
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in text]  # gets rid of the punctuation
    # https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
    text = [word for word in text if word.isalpha()]
    text = ' '.join(text)
    return text

def tokenize_with_keras(training, testing):
    tokenizer = Tokenizer(num_words = 5000) #look into what the propper num words is
    tokenizer.fit_on_texts(training)
    training = tokenizer.texts_to_sequences(training)
    testing = tokenizer.texts_to_sequences(testing)
    vocab_size = len(tokenizer.word_index) +1
    length = 100 #we are cutting off our reviews at 100 words and padding them to be 100, we can change this later if we want
    training = pad_sequences(training, padding='post', maxlen=length)
    testing = pad_sequences(testing, padding='post', maxlen=length)
    return training, testing, vocab_size, tokenizer

def embedding_matrix(vocab_size, tokenizer):
    embeddings_dictionary = dict()
    glove_file = open('../glove.6B.100d.txt', encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embed_vector = embeddings_dictionary.get(word)
        if embed_vector is not None:
            embedding_matrix[index] = embed_vector
    return embedding_matrix
#https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/

def get_tokenized_reviews(reviews, byWord=False):
    reviews_tokenized = []
    for review in reviews:    
        #Tokenize by sentence or word based on bool
        if byWord:
            reviews_tokenized.append(word_tokenize(review))
        else:
            reviews_tokenized.append(sent_tokenize(review))
    return reviews_tokenized

def main():
    reviews, ratings = get_data('review.json')
    #tokenized_reviews = get_tokenized_reviews(reviews)
    #print(reviews)

    # To get sent_tokenize to work, In python terminal: 
    # >>> import nltk
    # >>> nltk.download('punkt')
#    reviews_tokenized = tokenize_reviews(reviews)
    #reviews, ratings = get_data('yelp_dataset/review.json')

if __name__ == '__main__':
    main()