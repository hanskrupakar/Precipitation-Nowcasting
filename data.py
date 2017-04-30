import numpy as np
from numpy import genfromtxt
import re
import sys
import h5py
import math
import argparse

def clean(name, pattern=r'[^\d,\s\t\n\r/:]+'):

	with open(name, 'r+') as f:
		text = re.sub(r'[^\d,\s\t\n\r/:]+', '', f.read())
		f.seek(0)
		f.write(text)	

def preprocess(name, method='zscore'):

	data15 = genfromtxt(name, delimiter=',')[:, 1:17] #ignore datatime and last feature because of too many missing values

	if len(np.argwhere(np.isnan(data15))):
		print 'Contains illegal characters'	
	
	#Convert Cumulative values into actual features
	for i in xrange(np.shape(data15)[0]-1, 0, -1):
	
		data15[i][12] = data15[i][12] - data15[i-1][12] if data15[i][12]>=data15[i-1][12] else data15[i][12]
		data15[i][13] = data15[i][13] - data15[i-1][13] if data15[i][13]>=data15[i-1][13] else data15[i][13]
		data15[i][15] = data15[i][15] - data15[i-1][15] if data15[i][15]>=data15[i-1][15] else data15[i][15]
	
	#Cumulative initial timestep values set to 0
	data15[0][12]=data15[0][13]=data15[0][15]=0
	
	#precipitation(inches to mm)
	data15[:,15] *= 25.4 
	
	if method=='zscore':
		# z-score norm
		mean = np.mean(data15[:,:15], axis = 0)
		std = np.std(data15[:,:15], axis = 0)
		data15[:,:15] = (data15[:,:15]-mean)/std
	
		prec = [np.sum(data15[i:i+24, 15]) for i in xrange(len(data15)-24)]
		mean = np.append(mean, np.mean(np.array(prec), axis=0)) 
		std = np.append(std, np.std(np.array(prec), axis=0))
		
		return data15, np.array(mean), np.array(std)
	
	else:
		#min-max norm between -1 and 1
		minimum = np.amin(data15[:,:15], axis = 0)
		maximum = np.amax(data15[:,:15], axis = 0)
		data15[:,:15] = 2*((data15[:,:15]-minimum)/(maximum-minimum))-1

		prec = [np.sum(data15[i:i+24, 15]) for i in xrange(len(data15))]
		minimum = np.append(minimum, np.amin(np.array(prec), axis=0)) 
		maximum = np.append(maximum, np.amax(np.array(prec), axis=0))

		return data15, np.array(minimum), np.array(maximum)

def create_batches(data15, batch_size, a, b, method='zscore'):
	
	X, Y, x_batch, y_batch = [], [], [], []
	
	total = np.shape(data15)[0] - 47
	total = total - total%batch_size
	
	for i in xrange(total):	
		
		if len(x_batch)==batch_size:
			#create batch if size is batch_size
			X.append(np.array(x_batch))
			if method=='zscore':
				Y.append(np.array(np.reshape((y_batch-a)/b, (-1,1)))) 
			else:
				Y.append(np.array(np.reshape(2*(y_batch-a)/(b-a)-1, (-1,1))))
			x_batch=[]
			y_batch=[]
		
		#take 24 15min readings for 6 hour cumulative precipitation results
		x_batch.append(data15[i:i+24, :15])
		y_batch.append(np.sum(data15[i+24:i+48, 15]))
	
	return np.array(X), np.array(Y)
		
def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-batch_size', '--batch_size', default=32, required=False)
	parser.add_argument('-method', '--method', default="minmax", required=False)
	args = parser.parse_args()
	
	#clean('Agrimet 15min.csv')
	
	data, a, b = preprocess('Agrimet 15min.csv', args.method)
	
	x_batches, y_batches = create_batches(data, args.batch_size, a[15], b[15], args.method)
	
	ntrain, ntest = 0.9, 0.1
	
	tno = int(math.ceil(np.shape(x_batches)[0]*ntrain))
	testno = np.shape(x_batches)[0]-tno
	
	print 'TRAIN SET (#/batch=%d):'%args.batch_size ,tno, 'batches\nTEST SET (#/batch=%d):'%args.batch_size, testno, 'batches'
	
	trainx, testx = x_batches[:tno], x_batches[tno:]
	trainy, testy = y_batches[:tno], y_batches[tno:]
	
	with h5py.File('%s_full_dataset.h5'%args.method, 'w') as f:
		f.create_dataset('train/x', data=trainx)
		f.create_dataset('train/y', data=trainy)
		f.create_dataset('test/x', data=testx)
		f.create_dataset('test/y', data=testy)
		
		if args.method=='zscore':
			f.create_dataset('mean', data=a)
			f.create_dataset('std', data=b)
		else:
			f.create_dataset('min', data=a)
			f.create_dataset('max', data=b)
			
if __name__ == "__main__":	
	main()
