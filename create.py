import os
import numpy as np
import datetime
import time
import itertools
import math
import h5py
from data import preprocess
import random

def extract_into_array(text='Agrimet 15min.csv', imgdir='RADAR DATA/Dataset'):

	data15, minimum, maximum = preprocess(text)

	text = [[datetime.datetime.strptime(x, '%m/%d/%Y %H:%M'), y] #convert to datetime every date and time values from CSV file
			for x, y in #choose all the values
				zip(np.genfromtxt(text, delimiter=',', dtype='string')[:,0], data15)] #datetime row

	images = [[datetime.datetime.strptime( #convert to datetime every date and time values from available image folders
				x[0].split('/')[-1][21:29], '%Y%m%d'), x[0]] #extract dirname alone too
					 for x in os.walk(imgdir) if x[0]!=imgdir] #over all folders
	images.sort(key=lambda x: x[1])

	return text, images, minimum, maximum

def create_dataset(text, images, minimum, maximum):

	n, X, X_img, Y = 0, [], [], []

	prev_date = datetime.datetime.min #Dummy day for initialization purposes to facilitate comparison inside loop

	for x, directory in images: #for all the days (in ascending order) present in radar dataset
	
		#find textual features for the same days as present in the radar dataset
		while text[n][0].date()!=x.date():
			n+=1
		start=n
		while text[n][0].date()==x.date():
			n+=1
		end=n
	
		#Create sorted (ascending order) arrays to represent the images and the text alongwith their datetime stamps
		day_images = [[datetime.datetime.strptime(y[19:31], '%Y%m%d%H%M'), directory+'/'+y] for y in os.listdir(directory)]
		day_text = text[start:end]
	
		c, track,  = 0, 0
	
		if x.date()-prev_date.date()==datetime.timedelta(days=1):
			X_day, X_img_day, Y_day = X_day[-23:], X_img_day[-23:], Y_day[-23:]
		else:
			X_day, X_img_day, Y_day = [], [], []
	
		if day_images: #Some directories are empty (don't contain the 3h Precip. accumulation feature images at all)
			for dt, feat in day_text: #Assign images from the directory to every timestamp in textual data
			
				if c!=0:
					c-=1 #account for multiple 15 min timestamps having the same radar image
			
				#find position c where dt is in between 2 time intervals of radar data ([c-1] and [c])
				while dt>day_images[c][0]:
					c+=1
					if c==len(day_images)-1:
						break
			
				#calculate cumulative precipitation's min-max norm for every [i:i+24] features as sum(precip[i+24:i+48]))
				y = 2*(np.sum(np.array([prec[15] for _, prec in text[start+track+24:start+track+48]]))-minimum[15])/(maximum[15]-minimum[15])-1
			
				X_day.append(feat[:15])
				Y_day.append(y)
			
				#Based on closest time difference heuristic				
				if dt-day_images[c-1][0]>day_images[c][0]-dt: 
					X_img_day.append(day_images[c][1]) 
				else:
					X_img_day.append(day_images[c-1][1])
				track+=1
		
			for i in xrange(len(X_day)-23):
				X.append(X_day[i:i+24])
				Y.append(Y_day[i])
				X_img.append(X_img_day[i:i+24])
		
		prev_date = x #for adding the last 24 values in a day if 2 consecutive days are present		

	return X, X_img, Y

def create_batches(X, X_img, Y, batch_size):

	index = range(np.shape(X)[0]) #shuffling indices array
	random.shuffle(index)

	size = 32-np.shape(X)[0]%batch_size #make dataset size batch-coherent

	X, X_img, Y = [X[i] for i in index], [X_img[i] for i in index], [Y[i] for i in index] #shuffle

	X, X_img, Y = np.vstack((X, X[-size:])), np.vstack((X_img, X_img[-size:])), np.vstack((np.reshape(Y, (len(Y), 1)), np.reshape(Y[-size:], (size,1)))) #make up batch coherency

	X, X_img, Y = np.reshape(X, (-1, batch_size, 24, 15)), np.reshape(X_img, (-1, batch_size, 24)), np.reshape(Y, (-1, batch_size, 1))

	# divide data into 90% training data and 10% testing data approximately
	tno = int(math.ceil(np.shape(X)[0]*0.9))
	testno = np.shape(X)[0]-tno 
	
	return tno, testno
	
def main():

	batch_size = 32
	t = time.time()
	
	text, images, minimum, maximum = extract_into_array()
	
	X, X_img, Y = create_dataset(text, images, minimum, maximum)
	
	tno, testno = create_batches(X, X_img, Y, batch_size)
	
	print 'TRAIN BATCHES (%d):'%batch_size ,tno, '\nTEST BATCHES (%d):'%batch_size, testno

	trainx, testx = X[:tno], X[tno:]
	trainy, testy = Y[:tno], Y[tno:]
	trainx_img, testx_img = X_img[:tno], X_img[tno:]

	with h5py.File('minmax_full_image.h5', 'w') as f:
			f.create_dataset('train/x', data=trainx)
			f.create_dataset('train/x_img', data=trainx_img)
			f.create_dataset('train/y', data=trainy)
			f.create_dataset('test/x', data=testx)
			f.create_dataset('test/x_img', data=testx_img)
			f.create_dataset('test/y', data=testy)
			f.create_dataset('min', data=minimum)
			f.create_dataset('max', data=maximum)

	print 'TIME TAKEN: ', '%1.3f'%(time.time()-t), 'seconds'
	
if __name__=='__main__':
	main()
