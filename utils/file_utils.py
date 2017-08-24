import pickle

PKL_FORMAT = '.pkl'


def save_data_to_file(data, labels, data_filename, labels_filename):
    with open(data_filename + PKL_FORMAT, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    with open(labels_filename + PKL_FORMAT, 'wb') as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)


def load_data_from_file(data_filename, labels_filename):
    with open(data_filename + PKL_FORMAT, 'rb') as f:
        data = pickle.load(f)

    with open(labels_filename + PKL_FORMAT, 'rb') as f:
        labels = pickle.load(f)

    return data, labels
