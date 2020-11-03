from keras.models import model_from_json
import numpy
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import backend as K


def run(text_input):  
	K.clear_session()
	f = open("model_architecture.json",'r+')
	json_string = f.read()
	f.close()
	model = model_from_json(json_string)

	model.load_weights('model_weights.h5')
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	max_words = 500

	word_index = imdb.get_word_index()
	
	s = numpy.array([word_index.get(word, 0) for word in text_input.split()])

	s = sequence.pad_sequences([s], maxlen=max_words)

	y_pred = model.predict_classes(s)

	if y_pred[0] == 1:
		return "Positive"
	else:
		return "Negative"
	K.clear_session()