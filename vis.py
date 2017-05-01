from matplotlib import pyplot as plt
import numpy as np

def file_export(name):

	v, lr, n = [], [], 1
	
	with open(name, 'r') as f:
		for line in f.readlines():
			n+=1
			if float(line.split(' ')[6])>400 and n>50:
				pass
			else:
				lr.append(float(line.split(' ')[11].strip()))
				v.append(float(line.split(' ')[6]))
				
	return v, lr

zxy, lzxy = file_export('log.txt')
zx, lzx = file_export('LOG.txt')
mmx, lmmx = file_export('minmax_norm_log.txt')
mmxy, lmmxy = file_export('minmax_full_log.txt')

plt.figure()
plt.ylim([0, 0.015])
plt.plot(np.arange(0, len(lmmxy)), lmmxy)
plt.plot(np.arange(0, len(lmmx)), lmmx)
plt.plot(np.arange(0, len(lzxy)), lzxy)
plt.plot(np.arange(0, len(lzx)), lzx)
plt.legend(['Min-Max (Actual Precipitation Values for error)', 'Min-Max on X values only', 'Z-Score normalized values for X and Y', 'Z-Score normalised X only'])

plt.savefig('lr_decay_0.01.png', bbox_inches='tight')
