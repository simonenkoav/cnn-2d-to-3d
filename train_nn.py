import create_train_dataset
from sklearn import linear_model
from sklearn.linear_model import Ridge
import utils.file_utils as file_utils
import numpy as np

count_of_object_points = create_train_dataset.count_of_object_points
data_filename = create_train_dataset.data_filename
labels_filename = create_train_dataset.labels_filename

train_data, train_labels = file_utils.load_data_from_file(data_filename, labels_filename)

print("train_data[1] = " + str(train_data[1]))
print("train_labels[1] = " + str(train_labels[1]))

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
train_labels = np.reshape(train_labels, (train_labels.shape[0], train_labels.shape[1] * train_labels.shape[2]))

print("-------")
print("train_data[10] = " + str(train_data[10]))
print("train_labels[10] = " + str(train_labels[10]))
print("train_data[-10] = " + str(train_data[-10]))
print("train_labels[-10] = " + str(train_labels[-10]))

quit()

clf = Ridge(alpha=1.0)
clf.fit(train_data, train_labels)
# a = clf.predict(np.array([0.53288267, -0.53081549,
#                  -0.54942103, -0.72761004,
#                  0, 0,
#                  0.38432409, 0.22969993]).reshape(1, -1))
#
# print("\n_______")
# print("prediction = " + str(a))

a = clf.predict(np.array([10, -4,
                          12, 7,
                          1, -1,
                          -10, 0]).reshape(1, -1))

print("\n_______")
print("prediction = " + str(a))

print("\nLASSO")
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(train_data, train_labels)

a = lasso.predict(np.array([0.53288267, -0.53081549,
                 -0.54942103, -0.72761004,
                 0, 0,
                 0.38432409, 0.22969993]).reshape(1, -1))

print("prediction 1 = " + str(a))

a = lasso.predict(np.array([10, -4,
                          12, 7,
                          1, -1,
                          -10, 0]).reshape(1, -1))

print("prediction 2 = " + str(a))


