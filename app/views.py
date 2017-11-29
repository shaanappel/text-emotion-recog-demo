from flask import render_template
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash

import rnn
from rnn import getScores
from app import app

import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

import pickle


app.config.update(dict(
    DEBUG=True,
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default'
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

model = None
tokenizer = None


@app.route('/', methods=['GET', 'POST'])
def index():
	global model
	global tokenizer

	error = None
	if request.method == 'POST':
		if request.form['inputTextarea'] != "":
			inputText = request.form['inputTextarea']
			session['inputText'] = inputText
			print(session.get('inputText', "No input text"))
			predictions, sentences = getScores(inputText, model, tokenizer)
			session['predictions'] = predictions
			session['sentences'] = sentences

	if model == None:
		# load saved model
		model_filename = 'rnn_demo_model.hdf5'
		print("loading model")
		model = load_model(model_filename)

		print("model loaded")
		print("loading tokenizer")

		#getTokenizer
		with open('tokenizer.pickle', 'rb') as handle:
			tokenizer = pickle.load(handle)

	fullText = session.get('inputText', "No input text")
	predictions = session.get('predictions', [])
	sentences = session.get('sentences', [])
	preds = zip(predictions, sentences)

	return render_template("index.html", shownText = fullText,  predictions = preds)


@app.route('/about')
def about():
    return render_template("about.html")
