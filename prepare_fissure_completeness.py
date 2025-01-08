import json
import os

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch
from lungmask import LMInferer

from constants import KEYPOINT_CNN_DIR
from data_processing.datasets import LungData
from data_processing.datasets import normalize_img, PointDataset
from data_processing.keypoint_extraction import compute_keypoints
from data_processing.point_features import compute_point_features
from data_processing.surface_fitting import pointcloud_surface_fitting, o3d_mesh_to_labelmap
from models.access_models import get_point_seg_model_class_from_args
from utils.general_utils import new_dir, create_o3d_mesh, mask_out_verts_from_mesh, remove_all_but_biggest_component, \
    kpts_to_world
from utils.sitk_image_ops import resample_equal_spacing, sitk_image_to_tensor

RESAMPLE_SPACING = 1.5


def preprocess_img(sitk_img, resample_spacing=RESAMPLE_SPACING):
    img_resample = resample_equal_spacing(sitk_img, target_spacing=resample_spacing)
    img_tensor = sitk_image_to_tensor(img_resample).float().unsqueeze(0).unsqueeze(0)
    img_norm = normalize_img(img_tensor)
    return img_norm


def preprocess_mask(mask_img, resample_spacing=RESAMPLE_SPACING):
    mask_resample = resample_equal_spacing(mask_img, target_spacing=resample_spacing, use_nearest_neighbor=True)
    mask_tensor = sitk_image_to_tensor(mask_resample).long().unsqueeze(0).unsqueeze(0)
    return mask_tensor


# def get_cnn_keypoints(cv_dir, input_img, resample_spacing=RESAMPLE_SPACING, lung_mask=None, device='cuda:0'):
#     input_img = input_img.to(device)
#     all_kps = []
#     for fold_nr in range(5):
#         model = LRASPP_MobileNetv3_large_3d.load(os.path.join(cv_dir, f'fold{fold_nr}', 'model.pth'), device=device)
#         model.eval()
#         model.to(device)
#
#         with torch.no_grad():
#             out = model.predict_all_patches(input_img)
#
#         # find predicted fissure points
#         fissure_points = out.argmax(1).squeeze() != 0
#
#         # apply lung mask
#         if lung_mask is not None:
#             fissure_points = torch.logical_and(fissure_points, lung_mask)
#
#         # nonzero voxels to points
#         kp = torch.nonzero(fissure_points) * resample_spacing
#         kp = kp.long().cpu()
#         all_kps.append(kp)
#
#     return all_kps


def get_datasets_for_all_folds(sample_points, kp_folder, image_folder, patch_feat):
    base_ds = PointDataset(sample_points, kp_mode='cnn', folder=kp_folder, image_folder=image_folder,
                           patch_feat=patch_feat, do_augmentation=False, only_val_data=True)
    return [base_ds.split_data_set(None, fold)[1] for fold in range(5)]


RESULTS_DIR = os.path.join('..', 'FissureCompleteness', 'results')
SEQUENCES = ('b31f', 'b70f')
COMPLETNESS_DATA_PATH = '../FissureCompleteness/TestDataRename'


if __name__ == '__main__':
    # settings
    use_smooth_imgs = True
    model_dir = '../fissure-segmentation/results/DGCNN_seg_cnn_image'
    with open(os.path.join(model_dir, 'commandline_args.json'), 'r') as f:
        args = json.load(f)
    kp_dir = new_dir('..', 'FissureCompleteness', 'point_data')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # create missing lung masks and dummy fissure masks
    ds = LungData(COMPLETNESS_DATA_PATH)

    lm = LMInferer()
    for i, (case, sequence) in enumerate(ds.ids):
        img = ds.get_image(i)

        # create lung mask if it doesn't exist yet
        try:
            ds.get_lung_mask(i)
        except IndexError:
            print(f'Creating lung mask for case {case}, {sequence}...')
            lung_mask = lm.apply(img)
            lung_mask = sitk.GetImageFromArray(lung_mask)
            lung_mask.CopyInformation(img)
            sitk.WriteImage(lung_mask, os.path.join(COMPLETNESS_DATA_PATH, f'{case}_mask_{sequence}.nii.gz'))

        # create dummy fissure mask if it doesn't exist yet
        if ds.get_fissures(i) is None:
            print(f'Creating dummy fissure mask for case {case}, {sequence}...')
            dummy_fissure = sitk.Image(ds.get_image(i).GetSize(), sitk.sitkUInt8)
            dummy_fissure.CopyInformation(img)
            sitk.WriteImage(dummy_fissure, os.path.join(COMPLETNESS_DATA_PATH, f'{case}_fissures_{sequence}.nii.gz'))

        # create dummy regularized fissure mask if it doesn't exist yet
        if ds.get_regularized_fissures(i) is None:
            print(f'Creating dummy regularized fissure mask for case {case}, {sequence}...')
            dummy_fissure = sitk.Image(ds.get_image(i).GetSize(), sitk.sitkUInt8)
            dummy_fissure.CopyInformation(img)
            sitk.WriteImage(dummy_fissure, os.path.join(COMPLETNESS_DATA_PATH, f'{case}_fissures_poisson_{sequence}.nii.gz'))

    # reload dataset
    ds = LungData(COMPLETNESS_DATA_PATH)
    point_ds = PointDataset(args['pts'], 'cnn', folder=kp_dir, image_folder=COMPLETNESS_DATA_PATH, patch_feat='image',
                            do_augmentation=False, only_val_data=True)

    # extract keypoints and features
    try:
        point_ds_per_fold = get_datasets_for_all_folds(2048, kp_dir, COMPLETNESS_DATA_PATH, 'image')
        if all(len(pd) == len(ds) for pd in point_ds_per_fold):
            run_extraction = False
        else:
            run_extraction = True
    except:
        point_ds_per_fold = [point_ds] * 5
        run_extraction = True

    if run_extraction:
        for i, (case, sequence) in enumerate(ds.ids):
            print(f'Extracting keypoints and features for case {case}, {sequence}...')

            # keypoints
            img, mask, fissures = ds.get_image(i), ds.get_lung_mask(i), ds.get_regularized_fissures(i)
            lobes = sitk.Image(img.GetSize(), sitk.sitkUInt8)
            lobes.CopyInformation(img)
            compute_keypoints(img=img, mask=mask, fissures=fissures, lobes=lobes, out_dir=kp_dir, case=case,
                              sequence=sequence, device=device, cnn_dir=KEYPOINT_CNN_DIR, src_data_dir=COMPLETNESS_DATA_PATH,
                              kp_mode='cnn')

            # features
            for fold in range(5):
                compute_point_features(ds, case, sequence, os.path.join(kp_dir, 'cnn', f'fold{fold}'), feature_mode='image', device=device)

        # reload point dataset
        point_ds_per_fold = get_datasets_for_all_folds(2048, kp_dir, COMPLETNESS_DATA_PATH, 'image')

    # run segmentation network
    for fold in range(5):
        print(f'Running segmentation network for fold {fold}...')
        fold_result_dir = os.path.join(RESULTS_DIR, f'fold{fold}')
        mesh_dir = new_dir(fold_result_dir, 'meshes')
        label_dir = new_dir(fold_result_dir, 'labels')
        plot_dir = new_dir(fold_result_dir, 'plots')
        model_class = get_point_seg_model_class_from_args(os.path.join(model_dir, 'commandline_args.json'))
        model = model_class.load(os.path.join(model_dir, f'fold{fold}', 'model.pth'), device=device)
        model.eval()
        model.to(device)
        point_ds = point_ds_per_fold[fold]

        # forward dataset as one batch
        points = torch.stack([point_ds[i][0] for i in range(len(point_ds))]).to(device)
        coords = points[:, :3]
        with torch.no_grad():
            all_labels_pred = model.predict_full_pointcloud(points, sample_points=args['pts'], n_runs_min=50)

        all_labels_pred = all_labels_pred.argmax(1)

        for i, (case, sequence) in enumerate(ds.ids):
            img, mask_img = ds.get_image(i), ds.get_lung_mask(i)
            mask_tensor = sitk_image_to_tensor(mask_img).bool()

            # convert coords to world space
            spacing = torch.tensor(point_ds.spacings[i], device=device)
            shape = torch.tensor(point_ds.img_sizes_index[i][::-1], device=device) * spacing.flip(0)
            pts = kpts_to_world(coords[i].transpose(0, 1), shape)  # points in millimeters

            # only current case labels
            labels_pred = all_labels_pred[i]

            meshes_predict = []
            for j in range(model.num_classes - 1):  # excluding background
                label = j + 1
                try:
                    depth = 6
                    mesh_predict = pointcloud_surface_fitting(
                        pts[labels_pred.squeeze() == label].cpu().numpy().astype(float),
                        crop_to_bbox=True, depth=depth)

                except ValueError as e:
                    # no points have been predicted to be in this class
                    print(e)
                    mesh_predict = create_o3d_mesh(verts=np.array([]), tris=np.array([]))

                # post-process surfaces
                mask_out_verts_from_mesh(mesh_predict, mask_tensor, spacing)  # apply lung mask
                right = label > 1  # right fissure(s) are label 2 and 3
                remove_all_but_biggest_component(mesh_predict, right=right,
                                                 center_x=shape[2] / 2)  # only keep the biggest connected component

                meshes_predict.append(mesh_predict)

                # write out meshes
                o3d.io.write_triangle_mesh(os.path.join(mesh_dir, f'{case}_fissure{label}_pred_{sequence}.obj'),
                                           mesh_predict)

            # write out label images (converted from surface reconstruction)
            # predicted labelmap
            labelmap_predict = o3d_mesh_to_labelmap(meshes_predict, shape=point_ds.img_sizes_index[i][::-1],
                                                    spacing=point_ds.spacings[i])
            label_image_predict = sitk.GetImageFromArray(labelmap_predict.numpy().astype(np.uint8))
            label_image_predict.CopyInformation(mask_img)
            sitk.WriteImage(label_image_predict, os.path.join(label_dir, f'{case}_fissures_pred_{sequence}.nii.gz'))

