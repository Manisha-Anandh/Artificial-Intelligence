from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

dataset = loadtxt(r"F:\AI\day12-diabetiesPrediction\trainData\diabetes_prediction_dataset.csv",delimiter=',') 
##print(dataset)
##gender,age,hypertension,heart_disease,Smoking,bmi,HbA1c_level,blood_glucose_level,diabetes-labels of dataset
##print(np.shape(dataset))

x = dataset[0:1000,0:8] #input
y = dataset[0:1000,8]   #output

##print("input",x)
##print("output",y)

model = Sequential()

model.add(Dense(12, input_dim=8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

#model train
model.fit(x,y,epochs = 10,batch_size = 10)

#evaluation
_, accuracy = model.evaluate(x,y)
print("Accuracy: %.2f"%(accuracy*100))

#save model in json file
model_json = model.to_json()
with open("model.json","w")as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")
print("Saved model to disk")

      







          
          
