import numpy as np
from numpy import genfromtxt
import re
import h5py
import math

def clean(name, pattern=r'[^\d,\s\t\n\r/:]+'):

	with open(name, 'r+') as f:
		text = re.sub(r'[^\d,\s\t\n\r/:]+', '', f.read())
		f.seek(0)
		f.write(text)	

def preprocess(name):

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
	
	#min-max norm between -1 and 1
	minimum = np.amin(data15[:,:15], axis = 0)
	maximum = np.amax(data15[:,:15], axis = 0)
	data15[:,:15] = 2*((data15[:,:15]-minimum)/(maximum-minimum))-1

	prec = [np.sum(data15[i:i+24, 15]) for i in xrange(len(data15))]
	minimum = np.append(minimum, np.amin(np.array(prec), axis=0)) 
	maximum = np.append(maximum, np.amax(np.array(prec), axis=0))

	return data15, np.array(minimum), np.array(maximum)

def create_batches(data15, batch_size, pmin, pmax):
	
	X=[]
	Y=[]
	x_batch = []
	y_batch = []
	prec=[]
	
	total = np.shape(data15)[0] - 23
	total = total - total%batch_size
	
	for i in xrange(total):	
		
		if len(x_batch)==batch_size:
			#create batch if size is batch_size
			X.append(np.array(x_batch))
			Y.append(np.array(np.reshape(2*(y_batch-pmin)/(pmax-pmin)-1, (-1,1))))
			x_batch=[]
			y_batch=[]
		
		#take 24 15min readings for 6 hour cumulative precipitation results
		x_batch.append(data15[i:i+24, :15])
		y_batch.append(np.reshape(np.sum(data15[i:i+24, 15]), (-1, 1)))
	
	return np.array(X), np.array(Y)
		
def main():
	
	batch_size=32
	
	#clean('Agrimet 15min.csv')
	
	data, minimum, maximum = preprocess('Agrimet 15min.csv')
	
	np.random.shuffle(data)
	
	x_batches, y_batches = create_batches(data, batch_size, minimum[15], maximum[15])
	
	ntrain, nval, ntest = 0.9, 0.05, 0.05
	
	tno = int(math.ceil(np.shape(x_batches)[0]*ntrain))
	vno = int(math.ceil(np.shape(x_batches)[0]*nval))
	testno = np.shape(x_batches)[0]-tno-vno
	
	print 'TRAIN BATCHES (%d):'%batch_size ,tno, '\nVAL BATCHES (%d):'%batch_size, vno, '\nTEST BATCHES (%d):'%batch_size, testno
	
	trainx, valx, testx = x_batches[:tno], x_batches[tno:tno+vno], x_batches[tno+vno:]
	trainy, valy, testy = y_batches[:tno], y_batches[tno:tno+vno], y_batches[tno+vno:]
	
	with h5py.File('minmax_full_dataset.h5', 'w') as f:
		f.create_dataset('train/x', data=trainx)
		f.create_dataset('train/y', data=trainy)
		f.create_dataset('val/x', data=valx)
		f.create_dataset('val/y', data=valy)
		f.create_dataset('test/x', data=testx)
		f.create_dataset('test/y', data=testy)
		f.create_dataset('min', data=minimum)
		f.create_dataset('max', data=maximum)
	
main()
