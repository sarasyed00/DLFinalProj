import os
import numpy as np
import tensorflow as tf
from preprocessing import *


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        #Hyperparameters
        self.batch_size = 1
        self.learning_rate = .01
        self.hidden_size_1 = 100
        self.hidden_size_2 = 100

        #Layers
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.dense_1 = tf.keras.layers.Dense(self.hidden_size_1, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(self.hidden_size_2, activation='softmax')


    def call(self, inputs):
        layer1_output = self.dense_1(inputs)
        layer2_ouput = self.dense_2(layer1_output)


        return layer2_ouput

    def loss(self, probabilities, ratings):
        return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ratings, logits=probabilities))

    def accuracy(self, probabilities, ratings):
        pass

def train(model, train_rewards, train_ratings):
    
    #TO DO: Shuffle Inputs
    
    for i in range(0, len(train_rewards), model.batch_size):
        rewards_batch = train_rewards[i : i + model.batch_size]
        ratings_batch = train_ratings[i : i + model.batch_size]

        with tf.GradientTape() as tape:
            probabilities = model.call(rewards_batch)
            loss = model.loss(probabilities, train_ratings)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    return None

def test():
    pass

def main():
    model = Model()

    #TO DO: Figure out split for train and test
    train_rewards, train_ratings = get_data('./yelp_dataset/review.json')
    



if __name__ == '__main__':
	main()