import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot

import numpy as np


class preTrainNeuralNetwork:
	def __init__(self, wih_filename, who_filename):
		self.wih = numpy.load(wih_filename)
		self.who = numpy.load(who_filename)
		self.activation_function = lambda x: scipy.special.expit(x)
	def query(self, inputs_list):
		inputs = numpy.array(inputs_list, ndmin=2).T
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		return final_outputs
        
inputs_arr = matplotlib.pyplot.imread('number.png')  
inputs_list = inputs_arr.flatten()
nn = preTrainNeuralNetwork('w_ih.npy', 'w_ho.npy')
b =  nn.query(inputs_list)
print (b)
print(numpy.argmax(b))	
