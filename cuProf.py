import sys
from subprocess import call

import numpy as np
import matplotlib.pyplot as plt

args = sys.argv[1:len(sys.argv)]
for i in range (0, len(args)-1):
   a = str(args[i])
   end = len(a)-1
   if a[end] == ',':
      a = a[:end]+' '+str(args.pop(i+1))
   args[i] = a

kernel = []
DtoH = []
HtoD = []
names = []

for a in args:
	#call('nvprof --profile-api-trace none --log-file out.dat ./'+a, shell=True)
	names.append(a) #check ret first
	f = open('out.dat', 'r')
	for i in range (0,4):
		f.readline()
	dth = 0.
	htd = 0.
	k   = 0.
	for line in f:
		if line[0] == '=' or len(line) == 1:
			break
		data = line.split()
		pct = float(data[0][:-1])
		name = ''.join(data[6:])
		if (name[1:5] == 'CUDA'):
			call = name[-5:-1]
			if call == 'HtoD':
				htd += pct
			elif call == 'DtoH':
				dth += pct
		else:
			k += pct
	HtoD.append(htd)
	DtoH.append(dth)
	kernel.append(k)
		#print name+', '+str(pct)
	f.close()

ind = np.arange(len(names))
width=0.3
p1 = plt.bar(ind, HtoD, width, color='r')
p2 = plt.bar(ind, DtoH, width, color='b', bottom=HtoD)
p3 = plt.bar(ind, kernel, width, bottom=HtoD, color='g')

plt.xticks(ind + width/2., names )
plt.yticks(np.arange(0,135,10))
plt.legend( (p1[0], p2[0], p3[0]), ('HtoD', 'DtoH', 'kernel') )

plt.show()
