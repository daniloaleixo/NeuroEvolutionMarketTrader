import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))



def plot_loss(train_loss, val_loss):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(train_loss, label='Train Loss')
	plt.plot(val_loss, label='Validation Loss')
	plt.legend()
	plt.show()
    
def plot_results(predicted_data, true_data):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	
	# Pad the list of predictions to shift it in the graph to it's correct start
	for i, data in enumerate(predicted_data):
			padding = [None for p in range(i * prediction_len)]
			plt.plot(padding + data, label='Prediction')
			plt.legend()
	plt.show()
    
def get_acc(predictions_denorm, last_candle, close, test_name="Test"):
	'''
	Make the predictions and evaluate how well the algorithm corrects predict upward
	or downward movements
	'''
	ground_truth = last_candle[:, 0] - close[:, 0]
	predictions_truth = last_candle[:, 0] - predictions_denorm
	pred_bool = np.array(predictions_truth > 0)
	y_bool = np.array(ground_truth > 0)
	acc_test = np.count_nonzero(np.array(pred_bool == y_bool)) / y_bool.shape[0]
	print(test_name, " Acc ", '%.2f' % (acc_test * 100), "%")

def get_accuracy(y, predictions):
	'''
	Give accuracy based on movement of detrend value
	'''
	y_bool = np.array(y < 0)
	predictions_bool = np.array(predictions[:, 0] < 0)
	equal = np.array(y_bool == predictions_bool)
	return np.sum(equal) / equal.shape


def flatten_predict_array(array):
	a = np.zeros((array.shape[1], array.shape[0] * array.shape[2]))
	for i in range(array.shape[1]):
		a[i] = array[:, i, :].reshape((array.shape[0] * array.shape[2], ))
	return a
