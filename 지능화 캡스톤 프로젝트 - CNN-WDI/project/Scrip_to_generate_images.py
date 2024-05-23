import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

df = pd.read_pickle('../CNN-WDI/data/LSWMD.pkl')

s = df.size

for i in range(0,s):	
	if df.trianTestLabel[i].size > 0:
		img = df.waferMap[i]
		trainTest = df.trianTestLabel[i][0][0]
		failure = df.failureType[i][0][0]
		if trainTest == "Training":
			loc = 'waferimages/training/'+str(failure)+'/'+str(i)+'.png'
		else:
			loc = 'waferimages/testing/'+str(failure)+'/'+str(i)+'.png'
		plt.imshow(img)
		plt.savefig(loc,bbox_inches='tight')
		plt.clf()
	if i % 100 == 0:
		print(str(i))
		collected = gc.collect()
		print("Collected: "+str(collected))
	i = i + 1
		