import xgboost as xgb
import utils.file_utils as file_utils
import create_train_dataset
import numpy as np

count_of_object_points = create_train_dataset.count_of_object_points
data_filename = create_train_dataset.data_filename
labels_filename = create_train_dataset.labels_filename

train_data, train_labels = file_utils.load_data_from_file(data_filename, labels_filename)

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
train_labels = np.reshape(train_labels, (train_labels.shape[0], train_labels.shape[1] * train_labels.shape[2]))

train_M = xgb.DMatrix(train_data, label=train_labels)

params = {'max_depth': 3, 'eta': 0.1, 'gamma': 10000, 'silent': 1, 'objective': 'reg:linear',
          'eval_metric': 'rmse'}

evallist = [(train_M, 'train')]

num_round = 3
bst = xgb.train(params.items(), train_M, num_round, evallist)
bst.save_model('model_XGB.model')


test = [[1, 1, -1, 1, -1, -1, 1, -1]]
test_M = xgb.DMatrix(test)
pr1 = bst.predict(test_M)
print("pr1 = " + str(pr1))
