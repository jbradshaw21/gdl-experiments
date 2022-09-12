import numpy as np


def numpy_data_loader(data, train_samples, test_samples, labels, features, sample_type):

    train_data, test_data, train_labels, test_labels = [None for _ in range(4)]
    for data_samples in (train_samples, test_samples):
        feature_dict = {}
        label_list = []
        cell_no = 0
        entry_list = []
        for data_sample in data_samples:
            data_no = 0
            data_sample_arr = None
            data_sample_arr_exists = False
            for well_idx, well in enumerate(data["WellId"], 0):
                if well == data_sample[0]:
                    if data["FieldNumber"][well_idx] == data_sample[1] or sample_type == 'wells':
                        feature_arr = None
                        arr_exists = False
                        for feature in features:
                            if arr_exists:
                                feature_arr = np.concatenate((feature_arr, np.array([data[feature][well_idx]])))
                            else:
                                feature_arr = np.array([data[feature][well_idx]])
                                arr_exists = True

                        feature_arr = np.reshape(feature_arr, (1, feature_arr.shape[0]))
                        if data_sample_arr_exists:
                            data_sample_arr = np.concatenate((feature_arr, data_sample_arr), axis=1)
                        else:
                            data_sample_arr = feature_arr
                            data_sample_arr_exists = True

                        cell_no += 1
                        data_no += 1

            if data_sample_arr_exists:
                if data_sample in entry_list:
                    feature_dict[data_sample] = np.concatenate((feature_dict[data_sample], data_sample_arr), axis=1)
                else:
                    feature_dict[data_sample] = data_sample_arr
                    entry_list.append(data_sample)

                label_list.append(labels[data_sample[0]])

        feature_list = feature_dict.values()
        if data_samples == train_samples:
            train_data = np.vstack(feature_list)
            train_labels = np.array(label_list)
        else:
            test_data = np.vstack(feature_list)
            test_labels = np.array(label_list)

    return train_data, test_data, train_labels, test_labels
