# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:22:57 2020

@author: Sandi
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn import metrics

from sklearn.neural_network import MLPRegressor
data=np.loadtxt("dataset.csv", delimiter=",")
print(data)

data= data[data[:, 2].argsort()]
print(data)
print(data[:, 3].shape)

day_=0
bin_=[]
averages=[]
for lat,long,day,patient in data:
    if day==day_:
       bin_.append(patient) 
        
    else:    
        day_+=1
        averages.append(np.max(np.array(bin_)))
        bin_ = []
        bin_.append(patient)

"""

plt.figure()
plt.plot(data[:, 3])
plt.show()
#plt.plot(averages)
plt.show()
plt.close()
"""
input_data = np.vstack((np.array(data[:, 0]).T, 
                        np.array(data[:, 1]).T, 
                        np.array(data[:, 2]).T))
#print(input_data)
#print(input_data)

model = pickle.load(open("0.9794139335796535-modelf9ff0a66-8719-456c-9ee8-84013974d09d.pickle", 'rb'))
result = model.predict(input_data.T)
print(result)

day_=0
bin_=[]
averages_pred=[]

new_results = []
for x in range(len(data)):
    new_results.append([data[x, 0], data[x, 1], data[x, 2], result[x]])

for lat,long,day,patient in new_results:
    if day==day_:
       bin_.append(patient) 
        
    else:    
        day_+=1
        averages_pred.append(np.max(np.array(bin_)))
        bin_ = []
        bin_.append(patient)

plt.figure()
plt.plot(data[:, 3], label='Real data')
plt.plot(result[:], label='Predicted data')
plt.xlabel("Instances")
plt.ylabel("Number of cases")
plt.grid()
plt.legend()
plt.title("Model of recovered patients comparison to real data")
plt.savefig('recovered-data.pdf', dpi=1000, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format='pdf',
        transparent=False, bbox_inches = "tight", pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()
plt.plot(averages, label='Real data trend')
plt.plot(averages_pred, label='Predicted data trend')
plt.xlabel("Days")
plt.ylabel("Number of cases")
plt.grid()
plt.legend()
plt.title("Model trend of recovered patients comparison to real data")
plt.text(43, 31800, "$R^2=0.97$", fontsize=8, bbox=dict(facecolor='none', edgecolor='black'))
plt.savefig('recovered-model.pdf', dpi=1000, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format='pdf',
        transparent=False, bbox_inches = "tight", pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()
plt.close()




