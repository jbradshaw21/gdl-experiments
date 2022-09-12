from data_loader import data_loader
from visualize_points import visualize_points
from PointNetLayer import PointNetLayer

import torch
from torch_geometric.datasets import GeometricShapes
from torch_geometric.transforms import SamplePoints
from torch_cluster import knn_graph
from torch.nn import Linear
from torch_geometric.nn import global_max_pool
import time
import numpy as np

num_of_points = 512
num_of_centres = 2
num_of_changes = 64
model_type = 'GNN'
stopping_criteria = 0.005

# Download simulated data
dataset = GeometricShapes(root='data/GeometricShapes')
torch.manual_seed(42)
dataset.transform = SamplePoints(num=num_of_points)


class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(2, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, dataset.num_classes)

    def forward(self, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)

        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # 5. Classifier.
        return self.classifier(h)


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = torch.nn.Linear(2, 32)
        self.conv2 = torch.nn.Linear(32, 32)
        self.classifier = torch.nn.Linear(32, dataset.num_classes)

    def forward(self, pos, batch):
        h = self.conv1(pos)
        h = h.relu()
        h = self.conv2(h)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # 5. Classifier.
        return self.classifier(h)


model = None
if model_type == 'GNN':
    model = PointNet()
elif model_type == 'CNN':
    model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.


def train(model, optimizer, loader):
    model.train()

    data_batch_cat = torch.LongTensor([])
    for i in range(10):
        data_batch = torch.tensor([i])
        data_batch = data_batch.repeat(num_of_points)
        data_batch_cat = torch.cat((data_batch_cat, data_batch))

    batch_no = 0
    batch_loader = torch.reshape(loader, shape=(4, 10 * loader.shape[1], loader.shape[2]))

    total_loss = 0
    for idx, batch in enumerate(batch_loader, 0):
        batch_no += 1
        data_y = torch.tensor([idx * 10 + index for index in range(10)])

        optimizer.zero_grad()  # Clear gradients.
        logits = model(batch, data_batch_cat)  # Forward pass.
        loss = criterion(logits, data_y)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.

        total_loss += loss.item() * 10

    return total_loss / len(batch_loader)


def test(model, loader, epoch, no_removed=0):
    model.eval()

    data_batch_cat = torch.LongTensor([])
    for i in range(10):
        data_batch = torch.tensor([i])
        data_batch = data_batch.repeat(num_of_points - no_removed * num_of_centres)
        data_batch_cat = torch.cat((data_batch_cat, data_batch))

    total_correct = 0
    batch_no = 0
    incorrect_index = []
    batch_loader = torch.reshape(loader, shape=(4, 10 * loader.shape[1], loader.shape[2]))
    for idx, batch in enumerate(batch_loader, 0):
        batch_no += 1
        data_y = torch.tensor([idx * 10 + index for index in range(10)])
        logits = model(batch, data_batch_cat)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data_y).sum())
        if epoch + 1 == 50:
            for index, tensor in enumerate(list(pred == data_y), 0):
                if tensor == torch.tensor(False):
                    incorrect_index.append((batch_no - 1) * 10 + index)

    return total_correct / len(loader), incorrect_index


def run_class(test_load, with_edges):

    train_loader = data_loader(data=GeometricShapes(root='data/GeometricShapes', train=True), change=None,
                               no_of_points=num_of_points, no_changes=num_of_changes, no_centres=num_of_centres)

    test_loader = None
    if test_load == 'attract':
        test_loader = data_loader(data=GeometricShapes(root='data/GeometricShapes', train=False), change='attract',
                                  no_of_points=num_of_points, no_changes=num_of_changes, no_centres=num_of_centres)
    elif test_load == 'remove':
        test_loader = data_loader(data=GeometricShapes(root='data/GeometricShapes', train=False), change='remove',
                                  no_of_points=num_of_points, no_changes=num_of_changes, no_centres=num_of_centres)
    elif test_load == 'normal':
        test_loader = data_loader(data=GeometricShapes(root='data/GeometricShapes', train=False), change=None,
                                  no_of_points=num_of_points, no_changes=num_of_changes, no_centres=num_of_centres)

    start_time = time.time()
    converged, epoch, prev_loss = [False, 0, []]
    while not converged:
        loss = train(model, optimizer, train_loader)
        if test_load == 'remove':
            test_acc, incorrect_index = test(model, test_loader, epoch, no_removed=num_of_changes)
        else:
            test_acc, incorrect_index = test(model, test_loader, epoch)
        print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}, Time:'
              f' {time.time() - start_time:.2f} seconds')

        prev_loss.append(loss)
        if len(prev_loss) >= 5:
            std = np.std(prev_loss[-5:])
            if std < stopping_criteria:
                converged = True

        epoch += 1

        # if epoch == 50:
        #     for i in incorrect_index:
        #         edge_index = knn_graph(loader[i], k=16)
                # if with_edges:
                #     visualize_points(loader[i], edge_index=edge_index)
                # else:
                #     visualize_points(loader[i])


run_class(test_load='remove', with_edges=True)
