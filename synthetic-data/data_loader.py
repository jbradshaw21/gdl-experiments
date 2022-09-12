import torch
from euclid_dist import euclid_dist
from torch_geometric.transforms import SamplePoints


def data_loader(data, change, no_of_points, no_changes, no_centres):
    data.transform = SamplePoints(num=no_of_points)

    if change == 'remove':
        no_removed = no_changes
    else:
        no_removed = 0

    dataset_pos = torch.zeros(size=(len(data), data[0].pos.shape[0] - no_removed * no_centres, 2))

    for shape_index in range(len(data)):
        shuffler = torch.randperm(data[shape_index].pos.shape[0])
        data[shape_index].pos = data[shape_index].pos[shuffler]

        pos = data[shape_index].pos
        pos = pos[:, :2]

        if change is not None:
            points_moved = torch.tensor([])
            for centre in range(no_centres):
                dist = torch.tensor([])
                for point in pos:
                    dist = torch.cat((dist, euclid_dist(pos[centre], point)))
                _, idx_dist = torch.sort(dist)
                if change == 'attract':
                    for c in range(no_centres):
                        idx_dist = idx_dist[idx_dist != c]
                    for point in points_moved:
                        idx_dist = idx_dist[idx_dist != point]
                change_index = torch.cat((idx_dist[:no_changes - 1], torch.tensor([centre])))
                sorted_change_index, _ = torch.sort(change_index, descending=True)
                for index in sorted_change_index:
                    if change == 'attract':
                        if index != centre:
                            pos[index][0] = pos[centre][0] + (pos[index][0] - pos[centre][0]) / 2
                            pos[index][1] = pos[centre][1] + (pos[index][1] - pos[centre][1]) / 2
                    elif change == 'remove':
                        pos = torch.cat((pos[:index], pos[index + 1:]))
                points_moved = torch.cat((points_moved, sorted_change_index))

        dataset_pos[shape_index] = pos

    return dataset_pos
