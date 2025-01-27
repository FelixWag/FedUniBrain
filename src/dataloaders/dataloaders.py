from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, RandSpatialCrop, LoadImage
from monai.data import ArrayDataset, DataLoader, ImageDataset
from glob import glob
import os
import numpy as np
import logging


def get_transformations(file_format: str, cropped_input_size: tuple):
    """
    :param file_format:
    :param cropped_input_size:
    :return: This function returns train image transforms, train segmentation transforms,
    validation image transforms and val. However, in our case, they are the same

    """
    crop_function = RandSpatialCrop(cropped_input_size, random_size=False)

    if file_format == '.nii.gz':
        logging.info('[INFO] Getting transformations for .nii.gz files')
        # These are all the transformations
        train_nii_imtrans = Compose(
            [
                EnsureChannelFirst(strict_check=True),
                crop_function,
                RandRotate90(prob=0.1, spatial_axes=(0, 2)),
            ]
        )

        val_nii_trans = Compose([
            EnsureChannelFirst()
        ])
        # This is train image transforms, train segmentation transforms, validation image transforms and val. However, in our case, they are the same
        return train_nii_imtrans, train_nii_imtrans, val_nii_trans, val_nii_trans
    else:
        raise ValueError(f'Not supported file format {file_format}')


def get_images_and_segmentations(data_config, dataset_name, shuffle=False, save_file_path=None, load_file_path=None):
    dataset = data_config.datasets[dataset_name]

    image_paths = sorted(glob(os.path.join(dataset['img_path'], dataset['filename_img'])))
    seg_paths = sorted(glob(os.path.join(dataset['seg_path'], dataset['filename_seg'])))

    if shuffle:
        # Shuffle the pairs
        paired_paths = list(zip(image_paths, seg_paths))
        np.random.shuffle(paired_paths)
        image_paths, seg_paths = zip(*paired_paths)

        # Convert them back to lists
        image_paths = list(image_paths)
        seg_paths = list(seg_paths)

    if save_file_path:
        with open(save_file_path, 'w') as f:
            for img_path, seg_path in zip(image_paths, seg_paths):
                f.write(f"{img_path},{seg_path}\n")

    if load_file_path:
        with open(load_file_path, 'r') as f:
            lines = f.readlines()
            image_paths, seg_paths = zip(*(line.strip().split(',') for line in lines))

    return image_paths, seg_paths


def create_dataloader(data_config, dataset_name: str, train_n_samples: int, num_workers: int, batch_size: int, save_file_path = None):
    # First get the the images and segmentation paths
    image_paths, seg_paths = get_images_and_segmentations(data_config=data_config, dataset_name=dataset_name,
                                                          save_file_path=save_file_path)
    train_imtrans, train_segtrans, val_imtrans, val_segtrans = get_transformations(file_format=data_config.file_extension,
                                                                                   cropped_input_size=data_config.cropped_input_size)

    # Split the data into training and validation
    train_imgs = image_paths[:train_n_samples]
    train_segs = seg_paths[:train_n_samples]

    val_imgs = image_paths[train_n_samples:]
    val_segs = seg_paths[train_n_samples:]

    train_dataset = ImageDataset(train_imgs, train_segs, transform=train_imtrans, seg_transform=train_imtrans)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                  pin_memory=False)

    val_dataset = ImageDataset(val_imgs, val_segs, transform=val_imtrans, seg_transform=val_segtrans)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=False)

    return train_dataloader, val_dataloader
