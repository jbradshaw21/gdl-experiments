import torch


def gnn_data_loader(data, train_samples, test_samples, labels, features, sample_type):

    train_dict, test_dict, train_labels, test_labels = [None for _ in range(4)]
    for data_samples in (train_samples, test_samples):

        loader_dict = {}
        loader_dict.update(dict.fromkeys(features, []))
        loader_dict["labels"] = []
        for data_sample in data_samples:
            tensor_exists = False
            feature_dict = {}
            feature_dict.update(dict.fromkeys(features, torch.tensor([])))
            for well_idx, well in enumerate(data["WellId"], 0):
                if well == data_sample[0]:
                    if data["FieldNumber"][well_idx] == data_sample[1] or sample_type == 'wells':
                        if tensor_exists:
                            for feature in features:
                                feature_dict[feature] = torch.cat((feature_dict[feature],
                                                                   torch.tensor([data[feature][well_idx]])))
                        else:
                            for feature in features:
                                feature_dict[feature] = torch.tensor([data[feature][well_idx]])
                            tensor_exists = True

            if tensor_exists:
                for feature in features:
                    loader_dict[feature] = loader_dict[feature] + [feature_dict[feature]]
                loader_dict["labels"].append(labels[data_sample[0]])

        if data_samples == train_samples:
            train_dict = loader_dict.copy()
            train_labels = loader_dict["labels"]
        else:
            test_dict = loader_dict.copy()
            test_labels = loader_dict["labels"]

    return train_dict, test_dict, train_labels, test_labels
