# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 08:46:57 2020

@author: pavan
"""

import pickle
from sklearn.externals import joblib
import os
os.chdir(r"D:\spoj\tweet sentiment\New folder")


loaded_model=joblib.load("model.pkl")
loaded_stop=joblib.load("stopwords.pkl")
loaded_vec=joblib.load("vectorizer.pkl")


print(loaded_model.predict(loaded_vec.transform(["i theft a new phone. its from Apple.. !"])))