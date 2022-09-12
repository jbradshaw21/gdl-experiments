import torch


def batch_data(features, data, labels, batch_size):

    batch_dict = {}
    batch_list_dict = {}
    batch_dict.update(dict.fromkeys(features, torch.tensor([])))
    batch_list_dict.update(dict.fromkeys(features, []))
    label_batch, label_batch_list, data_batch_cat, data_batch_cat_list = [[] for _ in range(4)]

    feature_batch = []
    feature_batch_list = []
    for feature in features:
        feature_batch.append(batch_dict[feature])
        feature_batch_list.append(batch_list_dict[feature])

    batch_exists, data_batch_cat_exists = [False for _ in range(2)]

    batch_counter = 0
    data_batch_counter = 0
    for idx, _ in enumerate(data[features[0]], 0):
        if batch_exists and (batch_counter % batch_size) == 0:
            for list_idx in range(len(features)):
                feature_batch_list[list_idx] = feature_batch_list[list_idx] + [feature_batch[list_idx]]

            label_batch_list.append(label_batch)
            batch_exists = False
        elif batch_exists:
            for batch_idx in range(len(features)):
                feature_batch[batch_idx] = torch.cat((feature_batch[batch_idx], data[features[batch_idx]][idx]))

            label_batch = torch.cat((label_batch, torch.tensor([labels[idx]])))
            batch_counter += 1
        else:
            for batch_idx in range(len(features)):
                feature_batch[batch_idx] = data[features[batch_idx]][idx]

            label_batch = torch.tensor([labels[idx]])
            batch_exists = True
            batch_counter += 1

        if data_batch_cat_exists and data_batch_counter % batch_size == 0:
            data_batch_cat_list.append(data_batch_cat)
            data_batch_cat_exists = False
        elif data_batch_cat_exists:
            data_batch = torch.tensor([data_batch_counter % batch_size])
            data_batch = data_batch.repeat(data[features[0]][idx].shape[0])
            data_batch_cat = torch.cat((data_batch_cat, data_batch))

            data_batch_counter += 1
        else:
            data_batch_cat = torch.tensor([])
            data_batch = torch.tensor([data_batch_counter % batch_size])
            data_batch = data_batch.repeat(data[features[0]][idx].shape[0])
            data_batch_cat = torch.cat((data_batch_cat, data_batch))

            data_batch_cat_exists = True
            data_batch_counter += 1

    if batch_exists:
        for list_idx in range(len(features)):
            feature_batch_list[list_idx] = feature_batch_list[list_idx] + [feature_batch[list_idx]]

        label_batch_list.append(label_batch)
    if data_batch_cat_exists:
        data_batch_cat_list.append(data_batch_cat)

    return feature_batch_list, label_batch_list, data_batch_cat_list
