from os.path import join
from os import listdir

from torch.utils import data
import numpy as np

from utils.batchgenerators.transforms import MirrorTransform, SpatialTransform


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".raw"])

class DatasetFromFolder3D_R(data.Dataset):
    def __init__(self, file_dir, shape, num_classes, is_aug=True):
        super(DatasetFromFolder3D_R, self).__init__()
        self.is_aug = is_aug
        self.image_filenames = [x for x in listdir(join(file_dir, "image")) if is_image_file(x)]
        self.file_dir = file_dir
        self.shape = shape
        self.num_classes = num_classes
        if is_aug:
            self.mirror_transform = MirrorTransform(axes=(0, 1, 2))
            self.spatial_transform = SpatialTransform(patch_size=shape,
                                                      patch_center_dist_from_border=np.array(shape)//2,
                                                      do_elastic_deform=True,
                                                      alpha=(0., 1000.),
                                                      sigma=(10., 13.),
                                                      do_rotation=True,
                                                      angle_x=(-np.pi / 9, np.pi / 9),
                                                      angle_y=(-np.pi / 9, np.pi / 9),
                                                      angle_z=(-np.pi / 9, np.pi / 9),
                                                      do_scale=True,
                                                      scale=(0.75, 1.25),
                                                      border_mode_data='constant',
                                                      border_cval_data=0,
                                                      order_data=1,
                                                      random_crop=True)
        else:
            self.spatial_transform = SpatialTransform(patch_size=shape,
                                                      patch_center_dist_from_border=np.array(shape) // 2,
                                                      do_elastic_deform=False,
                                                      do_rotation=False,
                                                      do_scale=False,
                                                      scale=(0.75, 1.25),
                                                      border_mode_data='constant',
                                                      border_cval_data=0,
                                                      order_data=1,
                                                      random_crop=True)

    def __getitem__(self, index):
        # 读取image和label
        image = np.fromfile(join(self.file_dir, "image", self.image_filenames[index]), dtype=np.uint16)
        n_pieces = int(image.shape[0] / (150 * 150))
        image = image.reshape(n_pieces, 150, 150)
        image = np.where(image < 0., 0., image)
        image = np.where(image > 2048., 2048., image)
        image = image.astype(np.float32)
        image = image / 2048.

        target_tumor_kidney = np.fromfile(
            join(self.file_dir, "label_tumor_kidney", self.image_filenames[index][:-7] + "Label_200.raw"),
            dtype=np.uint16)
        target_tumor_kidney = target_tumor_kidney.reshape(n_pieces, 150, 150)
        target_tumor_kidney = np.where(target_tumor_kidney == 100, 2, target_tumor_kidney)
        target_tumor_kidney = np.where(target_tumor_kidney == 300, 3, target_tumor_kidney)

        target_artery = np.fromfile(join(self.file_dir, "label_artery", self.image_filenames[index]), dtype=np.uint16)
        target_artery = target_artery.reshape(n_pieces, 150, 150)
        target_artery = np.where(target_artery > 0, 4, 0)

        target_vein = np.fromfile(join(self.file_dir, "label_vein", self.image_filenames[index]), dtype=np.uint16)
        target_vein = target_vein.reshape(n_pieces, 150, 150)
        target_vein = np.where(target_vein > 0, 1, 0)

        target_ureter = np.fromfile(join(self.file_dir, "label_ureter", self.image_filenames[index][:-16]+"2"+self.image_filenames[index][-15:-8]+"_Def_200.raw"), dtype=np.uint16)
        target_ureter = target_ureter.reshape(n_pieces, 150, 150)
        target_ureter = np.where(target_ureter > 0, 5, 0)

        target = np.concatenate([target_tumor_kidney[np.newaxis, :, :, :],
                                 target_vein[np.newaxis, :, :, :],
                                 target_artery[np.newaxis, :, :, :],
                                 target_ureter[np.newaxis, :, :, :]], axis=0)
        target = np.max(target, axis=0)


        image = image[np.newaxis, np.newaxis, :, :, :]
        target = target[np.newaxis, np.newaxis, :, :, :]
        data_dict = {"data": image,
                     "seg": target}
        if self.is_aug:
            data_dict = self.mirror_transform(**data_dict)

            data_dict = self.spatial_transform(**data_dict)
        else:
            data_dict = self.spatial_transform(**data_dict)

        image = data_dict.get("data")
        target = data_dict.get("seg")

        target = self.to_categorical(target[0, 0], self.num_classes)
        target = target.astype(np.float32)
        image = image[0]
        return image, target

    def reshape_img(self, image, shape):
        out = np.zeros(shape, dtype=np.float32)
        out[:image.shape[0], :, :] = image
        return out

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.image_filenames)

