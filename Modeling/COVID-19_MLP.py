# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:54:09 2020

@author: Sandi
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn import metrics


from sklearn.externals import joblib

import pickle
import uuid

descriptor = ("d", "r", "c")
filenames = [r'time_series_19-covid-Deaths.csv',
             r'time_series_19-covid-Recovered.csv',
             r'time_series_19-covid-Confirmed.csv']

params_dict = [{'hidden_layer_sizes':  [(4,4,4,4),(4,4),(4,4,3,3),(4,3,4),(10,10,10,10,10), (3,), (6,6,6,6), (4,4), (10,5,5,10), (6,), (12,12,12), (3,3,3), (6,6,6), (3,3,3,3,3), (12, 12, 6, 6, 3, 3)],
                'activation': ['relu','identity','logistic','tanh'],
                'solver': ['adam', 'lbfgs'],
                'learning_rate':['constant','adaptive','invscaling'],
                'learning_rate_init': [0.1,0.01,0.5, 0.00001],
                'alpha': [0.01,0.1,0.001, 0.0001],
                'max_iter': [10000]}]

print(zip(descriptor, filenames))

for descriptor_, fname in zip(descriptor, filenames):
    print("Working on:", fname)
    df = pd.read_csv(fname)
    #print(df)
    df.drop(columns=["Province/State","Country/Region"], inplace=True)
    
    #print(df)
    
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
    
    input_data = DATA[:,:-1]
    output_data = DATA[:, -1]
    
    input_train, input_test, output_train, output_test = train_test_split(input_data, output_data)
    
    
    train_start_time=datetime.now()
    clf = GridSearchCV(MLPRegressor(), params_dict, cv=3, n_jobs=-1, scoring='r2', verbose=10)
    clf.fit(input_train, output_train)
    train_end_time = datetime.now()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    uuid_=uuid.uuid4()
    file = open(descriptor_+"-"+str(uuid_)+"-results.txt", 'w')
    file.write("Data for: "+fname+"\n")
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
          
          file.write("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)+"\n")
    file.write("Total training time:"+str(train_end_time-train_start_time)+"\n")
    file.close()
    model_name = descriptor_+"-"+str(uuid_)+".pickle"
    joblib.dump(clf.best_estimator_, model_name)
