from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix

class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=2, patch_size=7000):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i_mesh = self.data_list.iloc[idx][0] #vtk file name

        # read vtk
        mesh = load(i_mesh)
        labels = mesh.celldata['Label'].astype('int32').reshape(-1, 1)

        # new way
        # move mesh to origin
        points = mesh.points()
        mean_cell_centers = mesh.center_of_mass()
        points[:, 0:3] -= mean_cell_centers[0:3]

        ids = np.array(mesh.faces())
        cells = points[ids].reshape(mesh.ncells, 9).astype(dtype='float')

        # customized normal calculation; the vtk/vedo build-in function will change number of points
        mesh.compute_normals()
        normals = mesh.celldata['Normals']

        # move mesh to origin
        barycenters = mesh.cell_centers() # don't need to copy
        barycenters -= mean_cell_centers[0:3]

        #normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
            cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
            cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
            barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
            normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))
        X = (X - np.ones((X.shape[0], 1)) * np.mean(X, axis=0)) / (np.ones((X.shape[0], 1)) * np.std(X, axis=0))
        # print(labels)
        Y = labels
    

        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int')
        # S1 = np.zeros([self.patch_size, self.patch_size], dtype='float')
        # S2 = np.zeros([self.patch_size, self.patch_size], dtype='float')

        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels > 0)[:, 0]  # tooth idx
        negative_idx = np.argwhere(labels == 0)[:, 0]  # gingiva idx

        num_positive = len(positive_idx)  # number of selected tooth cells
        # print('num_positive', num_positive )

        num_negative = self.patch_size - num_positive  # number of selected gingiva cells

        positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
        negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)

        selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))
        selected_idx = np.sort(selected_idx, axis=None)



        #
        X_train[:] = X[selected_idx, :]
        # print(selected_idx)
        # print('unique ', np.unique(Y[selected_idx, :]))
        # print('Y after ', Y )
        # print('here', Y[selected_idx[0]])
        Y_train[:] = Y[selected_idx]

        X_train = X_train.transpose(1, 0)

        Y_train = Y_train.transpose(1, 0)
        # print(Y_train)

        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train)}

        return sample

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = Mesh_Dataset('./train_list_1.csv')
    # print(dataset.__getitem__(1)['labels'])
    train_loader = DataLoader(dataset=dataset,
                              batch_size=3,
                              shuffle=True,
                              num_workers=0)
    for sample in train_loader:
        print(sample['cells'].shape)
