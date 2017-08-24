from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers as optimizers

import create_train_dataset
import numpy as np
import utils.file_utils as file_utils

batch_size = 153
num_classes = 3 * 4  # think about it, you don't have classes but have outputs, is this number means the same?
epochs = 16

count_of_object_points = create_train_dataset.count_of_object_points
data_filename = create_train_dataset.data_filename
labels_filename = create_train_dataset.labels_filename

train_data, train_labels = file_utils.load_data_from_file(data_filename, labels_filename)

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
train_labels = np.reshape(train_labels, (train_labels.shape[0], train_labels.shape[1] * train_labels.shape[2]))

model = Sequential()
model.add(Dense(2 * 4, activation='relu', input_shape=(2 * 4,)))
model.add(Dropout(0.2))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3 * 4, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=optimizers.sgd(),
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(train_data, train_labels))
score = model.evaluate(train_data, train_labels, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

model.save("model_keras.h5")


