import os
import numpy as np
import torch
import torch.nn as nn
from pointnet import *
import utils
import vedo

if __name__ == '__main__':
    
    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
      
    model_path = './models'
    # model_name = 'Mesh_Segementation_PointNet_15_classes_colossalai_200_epoch_with_argumentation_best.tar'
    # model_name = 'Mesh_Segementation_PointNett_2_classes_data3_3samples_2_best.tar'
    model_name = 'data3_3samples_checkpoint_2_with_augmentation.tar'
    # model_name = 'Mesh_Segementation_PointNett_2_classes_data3_3samples_2_with_augmentation_best.tar'
    
    # mesh_path = '/home/brucewu/data/OSU_Mesh_Tooth_Segmentation/GroundTruth/maxillary_surface_T0_np10000_bg_0_vtk'
    mesh_path = '/dev_data/dev/PointNet_Seg/data_preprocessed/samples'
    test_list = [1, 2, 3, 4, 5]
    test_mesh_filename = 'Sample_0{0}_d.vtp'
    test_path = './test'
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    
    num_classes = 2
    num_features = 15
          
    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet_Seg(num_classes=num_classes, channel=num_features).to(device, dtype=torch.float)
    
    # load trained model
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)
    
    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Testing
    dsc = []
    sen = []
    ppv = []
    
    print('Testing')
    model.eval()
    with torch.no_grad():
        for i_sample in test_list:
            
            print('Predicting Sample filename: {}'.format(test_mesh_filename.format(i_sample)))
            # read image and label (annotation)
            mesh = vedo.load(os.path.join(mesh_path, test_mesh_filename.format(i_sample)))
            # pre-processing: downsampling
            if mesh.ncells > 10000:
                print('\tDownsampling...')
                target_num = 10000
                ratio = target_num / mesh.ncells  # calculate ratio
                mesh_d = mesh.clone()
                mesh_d.decimate(fraction=ratio)
                predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)
            else:
                mesh_d = mesh.clone()
                predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)

            # move mesh to origin
            print('\tPredicting...')
            points = mesh_d.points()
            mean_cell_centers = mesh_d.center_of_mass()
            points[:, 0:3] -= mean_cell_centers[0:3]

            ids = np.array(mesh_d.faces())
            cells = points[ids].reshape(mesh_d.ncells, 9).astype(dtype='float32')

            # customized normal calculation; the vtk/vedo build-in function will change number of points
            mesh_d.compute_normals()
            normals = mesh_d.celldata['Normals']

            # move mesh to origin
            barycenters = mesh_d.cell_centers()  # don't need to copy
            barycenters -= mean_cell_centers[0:3]

            # normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
                cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
                cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
                barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
                normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

            X = np.column_stack((cells, barycenters, normals))

            X = (X-np.ones((X.shape[0], 1))*np.mean(X, axis=0)) / (np.ones((X.shape[0], 1))*np.std(X, axis=0))
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            
            # numpy -> torch.tensor
            X = torch.from_numpy(X).to(device, dtype=torch.float)
            
            tensor_prob_output = model(X).to(device, dtype=torch.float).detach()
            patch_prob_output = tensor_prob_output.cpu().numpy()
                
            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output predicted labels
            mesh2 = mesh_d.clone()
            mesh2.celldata['Label'] = predicted_labels_d
            vedo.write(mesh2, os.path.join(test_path, 'Sample_{}_deployed.vtp'.format(i_sample)))

            print('Sample filename: {} completed'.format(i_sample))
            # mesh2.to_vtp(os.path.join(test_path, 'Sample_{}_deployed.vtp'.format(i_sample)))
