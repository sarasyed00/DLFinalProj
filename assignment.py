import os
import numpy as np
import tensorflow as tf
from preprocessing import *
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding



class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_matrix):
        super(Model, self).__init__()

        #Hyperparameters
        self.batch_size = 5
        self.learning_rate = .01
        self.hidden_size_1 = 1
        self.hidden_size_2 = 100
        self.embedding_size = 100
        self.review_words_length = 100
        self.vocab_size = vocab_size
        self.rnn_size = 100
        self.num_classes = 2
        self.E = embedding_matrix

        #Layers
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.dense_1 = tf.keras.layers.Dense(self.hidden_size_1, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        self.lstm = tf.keras.layers.LSTM(self.rnn_size)
        #self.conv_1 = tf.keras.layers.Conv1D(128, 5, activation = 'relu')
        #self.max_pool = tf.keras.layers.MaxPool1D(pool_size=2,strides=1, padding='valid')
        self.embedding_layer = Embedding(self.vocab_size, self.embedding_size, weights = [embedding_matrix], input_length=self.review_words_length,
                                         trainable=True

    def call(self, inputs): #should output an array of batch_size x 2
        embedding = self.embedding_layer(inputs)
        #embedding = tf.nn.embedding_lookup(self.E, inputs)
        #convoluted = self.conv_1(embedding)
        #max_pooled = self.max_pool(convoluted)
        lstm= self.lstm(embedding)
        layer1_output = self.dense_1(lstm)
        layer2_output = self.dense_2(layer1_output)
        return layer2_output

    def loss(self, probabilities, ratings):
        return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ratings, logits=probabilities))

    def accuracy(self, probabilities, ratings):
        #print(probabilities)
        #print(ratings)
        accuracy = 0
        for x in range(len(ratings)):
            if tf.argmax(probabilities[x]) == ratings[x]:
                accuracy+=1
        return accuracy/len(ratings)

def train(model, train_rewards, train_ratings):
    
    #TO DO: Shuffle Inputs
    for i in range(0, len(train_rewards), model.batch_size):
        rewards_batch = train_rewards[i : i + model.batch_size]
        ratings_batch = train_ratings[i : i + model.batch_size]
        with tf.GradientTape() as tape:
            probabilities = model.call(rewards_batch)
            ratings_batch = np.reshape(np.array(ratings_batch), [model.batch_size, 1])
            loss = model.loss(probabilities, ratings_batch)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(loss)
    #print(model.accuracy(probabilities, ratings_batch))


def test(model,test_reviews, test_ratings):
    accuracies = []
    for i in range(0, len(test_reviews), model.batch_size):
        rewards_batch = test_reviews[i : i + model.batch_size]
        ratings_batch = test_ratings[i : i + model.batch_size]
        probabilities = model.call(rewards_batch)
        ratings_batch = np.reshape(np.array(ratings_batch), [model.batch_size, 1])
        loss = model.loss(probabilities, ratings_batch)
        accuracy = model.accuracy(probabilities, ratings_batch)
        accuracies.append(accuracy)
    return tf.reduce_mean(accuracies)



def main():
    reviews, ratings = get_data('review.json')
    rev_train, rev_test, rat_train, rat_test = train_test_split(reviews, ratings, test_size = .10, random_state=42)
    rev_train, rev_test, vocab_size, tokenizer = tokenize_with_keras(rev_train,rev_test)
    embed_matrix = embedding_matrix(vocab_size, tokenizer)
    #print(embed_matrix.shape)
    model = Model(vocab_size, embed_matrix)
    train(model, rev_train,rat_train)
    print(test(model, rev_test, rat_test))

    #print(rev_test)
    #print(vocab_size)

    #TO DO: Figure out split for train and test

    



if __name__ == '__main__':
	main()