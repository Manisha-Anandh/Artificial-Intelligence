from numpy import loadtxt
from keras.models import model_from_json

dataset = loadtxt(r"F:\AI\day12-diabetiesPrediction\trainData\diabetes_prediction_dataset.csv",delimiter=',')
x = dataset[0:1000,0:8]
y = dataset[0:1000,8]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.weights.h5")
print("Loaded model from disk")

predictions = model.predict(x)

for i in range(0,15):
    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
