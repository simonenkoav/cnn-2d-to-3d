from keras.models import load_model as load_model
import numpy as np

model = load_model("model_keras.h5")

pr = model.predict(np.array([[1, 1, -1, 1, -1, -1, 1, -1]]), 1)
print("pr1 = " + str(pr))
pr = model.predict(np.array([[-1, 1, -1, -1, 1, -1, 1, 1]]), 1)
print("pr2 = " + str(pr))