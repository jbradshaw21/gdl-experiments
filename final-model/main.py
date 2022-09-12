from pre_process_data import pre_process_data
from gnn_data_loader import gnn_data_loader
from numpy_data_loader import numpy_data_loader
from batch_data import batch_data
from cross_validation import cross_validation

import torch
from torch_cluster import knn_graph, radius_graph
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, global_max_pool

import time
import pandas as pd
import os
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

start_time = time.time()

model_used = 'svm'
data_set = 'Well'
sample_type = 'wells'
cv_folds = 5
graph_building_method = 'epsilon_ball'
graph_building_param = 275
batch_size = 16
stopping_criteria = 0.001
learning_rate = 0.0005
weight_decay = 0
# h_features = []
h_features = ['ObjectTotalAreaCh1', 'ObjectAvgAreaCh1', 'ObjectTotalIntenCh1', 'ObjectAvgIntenCh1',
              'ObjectTotalIntenPerObjectCh1', 'SpotCountCh2', 'SpotTotalAreaCh2', 'SpotAvgAreaCh2', 'SpotTotalIntenCh2',
              'SpotAvgIntenCh2', 'SpotTotalIntenPerSpotCh2', 'SpotCountPerObjectCh2', 'SpotTotalAreaPerObjectCh2',
              'SpotTotalIntenPerObjectCh2']
pos_features = ['X', 'Y']
cluster = False

random.seed(42)
torch.manual_seed(42)


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + len(pos_features), out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))

    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_i, pos_j):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]
        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.


class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PointNetLayer(2, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, 3)

    def forward(self, h, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = None
        if graph_building_method == 'knn':
            edge_index = knn_graph(pos, k=graph_building_param, batch=batch, loop=True)
        elif graph_building_method == 'epsilon_ball':
            edge_index = radius_graph(pos, r=graph_building_param, batch=batch, loop=True, max_num_neighbors=4096)

        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # 5. Classifier.
        return self.classifier(h)


device = 'cpu'
if 'COLAB_GPU' in os.environ or cluster:
    device = torch.device('cuda')

if model_used == 'attention_gnn':
    model = PointAttentionNet()
elif model_used == 'cnn':
    model = CNN()
else:
    model = PointNet()

criterion = torch.nn.CrossEntropyLoss().to(device)  # Define loss criterion.


def concat_features(h_feature_batch_loader, pos_feature_batch_loader, idx):
    h_features_cat, pos_features_cat = [None for _ in range(2)]
    concat_h = True
    for feature_batch_loader in (h_feature_batch_loader, pos_feature_batch_loader):
        features_cat = torch.tensor([])
        features_cat_exists = False
        for batch_loader in feature_batch_loader:
            batch_loader[idx] = torch.reshape(batch_loader[idx], (batch_loader[idx].shape[0], 1))
            if features_cat_exists:
                features_cat = torch.cat((features_cat, batch_loader[idx]), dim=1)
            else:
                features_cat = batch_loader[idx]
                features_cat_exists = True
        if concat_h:
            h_features_cat = features_cat
            concat_h = False
        else:
            pos_features_cat = features_cat

    return h_features_cat, pos_features_cat


def train(model, criterion, optimizer, features, data, labels):
    feature_batch_loader, label_batch_loader, data_batch_cat_loader = \
        batch_data(features=features, data=data, labels=labels, batch_size=batch_size)

    h_feature_batch_loader = feature_batch_loader[:len(features) - len(pos_features)]
    pos_feature_batch_loader = feature_batch_loader[len(features) - len(pos_features):]

    model.train()
    total_loss = 0
    for idx, _ in enumerate(data_batch_cat_loader, 0):
        h_features_cat, pos_features_cat = concat_features(h_feature_batch_loader=h_feature_batch_loader,
                                                           pos_feature_batch_loader=pos_feature_batch_loader, idx=idx)

        optimizer.zero_grad()  # Clear gradients.

        if model_used != 'cnn':
            logits = model(h=h_features_cat.float().to(device), pos=pos_features_cat.float().to(device),
                           batch=data_batch_cat_loader[idx].type(torch.LongTensor).to(device))  # Forward pass.
        else:
            logits = model(h=h_features_cat.float().to(device))

        loss = criterion(logits, label_batch_loader[idx].type(torch.LongTensor).to(device))  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * 10

    return total_loss / len(features[0])


def test(model, features, data, labels):
    feature_batch_loader, label_batch_loader, data_batch_cat_loader = \
        batch_data(features=features, data=data, labels=labels, batch_size=batch_size)

    h_feature_batch_loader = feature_batch_loader[:len(features) - len(pos_features)]
    pos_feature_batch_loader = feature_batch_loader[len(features) - len(pos_features):]

    model.eval()

    label_pred, label_true, prob = [[] for _ in range(3)]
    prob_exists = False

    for idx, _ in enumerate(data_batch_cat_loader, 0):
        h_features_cat, pos_features_cat = concat_features(h_feature_batch_loader=h_feature_batch_loader,
                                                           pos_feature_batch_loader=pos_feature_batch_loader, idx=idx)

        if model_used != 'cnn':
            logits = model(h=h_features_cat.float().to(device), pos=pos_features_cat.float().to(device),
                           batch=data_batch_cat_loader[idx].type(torch.LongTensor).to(device))  # Forward pass.
        else:
            logits = model(h=h_features_cat.float().to(device))

        label_pred += logits.argmax(dim=-1).tolist()
        label_true += label_batch_loader[idx].type(torch.LongTensor).tolist()
        if prob_exists:
            prob = np.concatenate((prob, np.array(logits.tolist())))
        else:
            prob = np.array(logits.tolist())
            prob_exists = True
        sum_of_rows = prob.sum(axis=1)
        prob = prob / sum_of_rows[:, np.newaxis]

    total_correct = 0
    for idx, label in enumerate(label_pred, 0):
        if label == label_true[idx]:
            total_correct += 1

    accuracy = total_correct / len(label_pred)
    roc_auc = roc_auc_score(np.array(label_true), prob, multi_class='ovr')

    return accuracy, roc_auc


def run_class():

    # Read csv data
    if 'COLAB_GPU' in os.environ:
        data = pd.read_csv(f'/content/drive/MyDrive/{data_set}.csv')
    else:
        data = pd.read_csv(f'C:/Users/jbrad/OneDrive/Documents/Thesis/Data/{data_set}.csv')

    data, sample_list, cell_count_dict, concentration_dict, label_dict = pre_process_data(data=data,
                                                                                          features=
                                                                                          h_features + pos_features,
                                                                                          sample_type=sample_type,
                                                                                          model_used=model_used,
                                                                                          data_set=data_set)
    print(f'Data pre-processed in {round(time.time() - start_time, 4)} seconds.')

    # no_activity_dict = {'4184': 0, '4951': 0, '1854': 0}
    # label_total_dict = {'4184': 0, '4951': 0, '1854': 0}
    # for idx, _ in enumerate(data['SpotCountCh2'], 0):
    #     well = data['WellId'][idx]
    #
    #     if well[-2:] not in ['15', '16', '17', '18']:
    #         if label_dict[well] == 0:
    #             label_total_dict['4184'] += 1
    #         elif label_dict[well] == 1:
    #             label_total_dict['4951'] += 1
    #         elif label_dict[well] == 2:
    #             label_total_dict['1854'] += 1
    #
    #         if data['SpotCountCh2'][idx] == 0:
    #             if label_dict[well] == 0:
    #                 no_activity_dict['4184'] += 1
    #             elif label_dict[well] == 1:
    #                 no_activity_dict['4951'] += 1
    #             elif label_dict[well] == 2:
    #                 no_activity_dict['1854'] += 1
    #
    # for label in ['4184', '4951', '1854']:
    #     print(f'Percentage of inactive cells for {label}: {no_activity_dict[label] / label_total_dict[label]}')
    #
    # compute variances
    for feature in h_features:
        feature_var = np.var(np.array([i for i in data[feature]]))
        print(f'Variance of {feature} is {feature_var}.')

    # compute covariances
    all_features = [[i for i in data[feature]] for feature in h_features]
    print(f'Covariance matrix is: {np.ma.cov(all_features)}')

    exit()

    cv_samples = cross_validation(cv_folds=cv_folds, sample_list=sample_list, label_dict=label_dict,
                                  concentration_dict=concentration_dict, cell_count_dict=cell_count_dict,
                                  sample_type=sample_type)
    print(f'CV completed in {round(time.time() - start_time, 4)} seconds.')

    training_dict = {}
    training_dict_empty = True
    accuracy_list, roc_auc_list = [[] for _ in range(2)]
    for i in range(cv_folds):
        train_samples = []
        for idx, samples in enumerate(cv_samples, 0):
            if idx != i:
                train_samples = train_samples + samples
        test_samples = cv_samples[i]
        random.shuffle(train_samples)

        class_features = h_features

        if model_used == 'gnn' or model_used == 'attention_gnn' or model_used == 'cnn':

            if model_used != 'cnn':
                class_features = class_features + pos_features

            train_data, test_data, train_labels, test_labels = gnn_data_loader(data=data,
                                                                               train_samples=train_samples,
                                                                               test_samples=test_samples,
                                                                               labels=label_dict,
                                                                               features=class_features,
                                                                               sample_type=sample_type)

            print(f'Fold {i + 1} loaded in {round(time.time() - start_time, 4)} seconds.')

            model = PointNet().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            test_acc, roc_auc, converged, prev_loss, epoch = [None, None, False, [], 0]
            while not converged:
                loss = train(model, criterion, optimizer, class_features, train_data, train_labels)
                test_acc, roc_auc = test(model, class_features, test_data, test_labels)
                print(f'Fold {i + 1}, Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f},'
                      f' ROC AUC Score: {roc_auc:.4f}, Time:'f' {time.time() - start_time:.2f} seconds')

                epoch += 1
                prev_loss.append(loss)
                if len(prev_loss) >= 5:
                    std = np.std(prev_loss[-5:])
                    if std < stopping_criteria:
                        converged = True

                if training_dict_empty:
                    training_dict['fold'] = [i + 1]
                    training_dict['epoch'] = [epoch]
                    training_dict['test_acc'] = [test_acc]
                    training_dict['roc_auc'] = [roc_auc]
                    training_dict['time'] = [time.time() - start_time]
                    training_dict_empty = False
                else:
                    training_dict['fold'] = training_dict['fold'] + [i + 1]
                    training_dict['epoch'] = training_dict['epoch'] + [epoch]
                    training_dict['test_acc'] = training_dict['test_acc'] + [test_acc]
                    training_dict['roc_auc'] = training_dict['roc_auc'] + [roc_auc]
                    training_dict['time'] = training_dict['time'] + [time.time() - start_time]

            print(f'Final metrics after {epoch:02d} epochs: Test Accuracy: {test_acc:.4f},'f' ROC AUC Score:'
                  f' {roc_auc:.4f}, Time:'f' {time.time() - start_time:.2f} seconds')

        else:

            train_data, test_data, train_labels, test_labels = numpy_data_loader(data=data,
                                                                                 train_samples=train_samples,
                                                                                 test_samples=test_samples,
                                                                                 labels=label_dict,
                                                                                 features=class_features,
                                                                                 sample_type=sample_type)

            print(f'Fold {i + 1} loaded in {round(time.time() - start_time, 4)} seconds.')

            clf = None
            if model_used == 'logistic_regression':
                clf = LogisticRegression(max_iter=100000, random_state=42)
            elif model_used == 'svm':
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            elif model_used == 'random_forest':
                clf = RandomForestClassifier(max_depth=2, random_state=42)
            clf.fit(train_data, train_labels)

            roc_auc = None
            test_acc = clf.score(test_data, test_labels)

            print(f'Fold {i + 1}, Test Accuracy: {round(test_acc, 4)}, Time:'f' {round(time.time() - start_time, 4)}'
                  f' seconds')

            if training_dict_empty:
                training_dict['fold'] = [i + 1]
                training_dict['test_acc'] = [test_acc]
                training_dict['time'] = [time.time() - start_time]
                training_dict_empty = False
            else:
                training_dict['fold'] = training_dict['fold'] + [i + 1]
                training_dict['test_acc'] = training_dict['test_acc'] + [test_acc]
                # training_dict['roc_auc'] = training_dict['roc_auc'] + [roc_auc]
                training_dict['time'] = training_dict['time'] + [time.time() - start_time]

        accuracy_list.append(test_acc)
        roc_auc_list.append(roc_auc)

    print(f'Model used: {model_used}')
    print(f'Graph-building features: {pos_features}')
    print(f'Additional features: {h_features}')
    print(f'Accuracy metric: {cv_folds}-fold CV')

    model_description = {'model_used': [model_used], 'pos_features': [pos_features],
                         'h_features': [h_features], 'cv_folds': [cv_folds]}

    if model_used == 'gnn' or model_used == 'attention_gnn' or model_used == 'cnn':
        print(f'Batch size: {batch_size}')
        print(f'Stopping criteria: {stopping_criteria}')
        print(f'Learning rate: {learning_rate}')
        print(f'Weight decay: {weight_decay}')

        model_description['batch_size'] = [batch_size]
        model_description['stopping_criteria'] = [stopping_criteria]
        model_description['learning_rate'] = [learning_rate]
        model_description['weight_decay'] = [weight_decay]

    if model_used == 'gnn' or model_used == 'attention_gnn':
        parameter = None
        if graph_building_method == 'knn':
            parameter = 'k'
        elif graph_building_method == 'epsilon_ball':
            parameter = 'epsilon'

        print(f'Graph-building method: {graph_building_method} with {parameter}={graph_building_param}')

        model_description['graph_building_method'] = [graph_building_method]
        model_description['graph_building_param'] = [graph_building_param]

    runtime = time.time() - start_time
    print(f'Final test accuracy after {cv_folds}-fold CV: {sum(accuracy_list) / cv_folds:.4f}')
    print(f'Runtime: {runtime:.2f} seconds')

    model_description['test_acc'] = sum(accuracy_list) / cv_folds
    model_description['time'] = runtime

    folder_name = f'exp_{time.time()}'
    directory = f'../../OneDrive/Documents/Thesis/Experiment Results/{folder_name}'
    os.makedirs(directory)
    if 'COLAB_GPU' in os.environ:
        directory = f'/content/drive/My Drive/Experiments'

    df_model = pd.DataFrame(model_description)
    df_model.to_csv(f'{directory}/model_out.csv', index=False)

    df_training = pd.DataFrame(training_dict)
    df_training.to_csv(f'{directory}/training_out.csv', index=False)


run_class()
