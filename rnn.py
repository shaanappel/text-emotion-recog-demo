import re
import csv
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import nltk
import unicodedata


# takes in array of sentences and featurizes it for rnn
def featurize(reviewText, tokenizer):
    print(tokenizer)
    print(reviewText)

    sequences = tokenizer.texts_to_sequences(reviewText)
    max_seq_len = 80
    X = sequence.pad_sequences(sequences, maxlen=max_seq_len)
    return X


def getScores(inputTxt, model, tokenizer):


    print(inputTxt)
    inputTxt = unicodedata.normalize('NFKD', inputTxt).encode('ascii','ignore')
    print(inputTxt)
    inputTxt = inputTxt.decode("utf-8")
    print(inputTxt)
    # break up into sentences
    reviewText = nltk.sent_tokenize(inputTxt)

    print(reviewText)

    # reviewText = ["To go there, the desired any and today I'm going to share with you my favorite cracks in the week."]

    # get featurized data
    X = featurize(reviewText, tokenizer)

    print("featurized succesfully")



    # predict on data
    class_probs = model.predict(X)
    class_preds = list(class_probs.argmax(axis=-1))
    class_probs = list(class_probs)

    print("predicted")

    labels = ["Anger", "Frustration", "Joy", "Sadness", "Neutral"]


    predictions = []
    for i in range(len(class_preds)):
        pred = class_preds[i]
        class_prob = list(class_probs[i])
        for j in range(len(class_prob)):
            class_prob[j] = round(float(class_prob[j]), 2)
        predictions.append((int(pred), class_prob))

    print(predictions)
    print(reviewText)



    return predictions, reviewText


