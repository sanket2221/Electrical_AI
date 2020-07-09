import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score , mean_squared_error

#data
R = np.arange(0.01,50,0.01,dtype=float).reshape(-1,1)
L = np.arange(0.01,50,0.01,dtype=float).reshape(-1,1)
C = np.arange(0.01,50,0.01,dtype=float).reshape(-1,1)
RLC = np.concatenate((R,L,C),axis=1)

w = 2*np.pi*50
Xl = w*L
Xc = 1/(w*C)
Z = np.sqrt(R**2 + (Xl-Xc)**2)



X_train, X_test, y_train, y_test = train_test_split(RLC, Z, test_size=0.25,random_state=0)

model = keras.Sequential([
    keras.layers.Dense(units=3,input_shape=(3,),activation='relu') ,
    keras.layers.Dense(3 ,activation='relu'),
    keras.layers.Dense(1,activation='relu'),
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=["mean_squared_error"])

model.fit(X_train, y_train ,epochs=50)

y_pred = model.predict(X_test)

error = mean_squared_error(y_test,y_pred)
