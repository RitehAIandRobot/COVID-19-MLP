# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:54:09 2020

@author: Sandi
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn import metrics

import pickle
import uuid

df = pd.read_csv(r'time_series_19-covid-Recovered.csv')
#print(df)

df.drop(columns=["Province/State","Country/Region"], inplace=True)

print(df)

DATA = np.array((0,0,0,0))

for i, j in df.iterrows():
    #print(i, j)
    latitude = j['Lat']
    longitude = j['Long']
    
    for k,l in j.iteritems():
        if k=='Lat':
            continue
        if k=='Long':
            continue
        date = datetime.strptime(k, '%m/%d/%y')
        day = date - date.strptime("01/22/20", '%m/%d/%y')
        days = day.days
        #print(days)
        temp = np.array([j["Lat"], j['Long'], days, l])
        #print(temp)
        DATA = np.vstack((DATA,temp))  
        
DATA = np.delete(DATA, 0,0)
np.random.shuffle(DATA)
print(DATA.shape)

np.savetxt("dataset.csv", DATA, delimiter=",")