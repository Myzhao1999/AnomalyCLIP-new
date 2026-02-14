import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['SDD']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    elif dataset_name == 'BrainMRI':
        obj_list = ['brain']
    elif dataset_name == 'headct':
        obj_list = ['brain']
    elif dataset_name == 'br35h':
        obj_list = ['br35h']
    elif dataset_name == 'clinic':
        obj_list = ['colon']
    elif dataset_name == 'endo':
        obj_list = ['colon']
    elif dataset_name == 'Kvasir':
        obj_list = ['colon']
    elif dataset_name =='mvtec-clinic':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood','colon']
    elif dataset_name=='DAGM':
        obj_list = ['Fabric1', 'Fabric2', 'Fabric3', 'Fabric4', 'Fabric5', 'Fabric7', 'Fabric8']
    elif dataset_name=='RSDD':
        obj_list = ['Metal15']
    elif dataset_name=='KSDD2':
        obj_list = ['Metal']
    elif dataset_name=='tn3k':
        obj_list = ['thyroid']
    #elif dataset_name=='MPDD':
    #    obj_list = ['bracket_black','bracket_brown','bracket_white','connector','metal_plate','tubes']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test', aug_rate=0.0, training=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.aug_rate = aug_rate
        self.training = training
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]
        self.meta_info = meta_info

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)

    def __len__(self):
        return self.length

    def _load_mask(self, data, img):
        if data['anomaly'] == 0:
            return Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')

        mask_path = data['mask_path']
        if (not mask_path) or os.path.isdir(os.path.join(self.root, mask_path)):
            # Some datasets only provide image-level labels.
            return Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')

        img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
        return Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

    def combine_img(self, cls_name):
        """
        From April-GAN: https://github.com/ByChelsea/VAND-APRIL-GAN
        Combine four images into one 2x2 image for augmentation.
        """
        img_info = random.sample(self.meta_info[cls_name], 4)

        img_ls = []
        mask_ls = []

        for data in img_info:
            img_path = os.path.join(self.root, data['img_path'])
            img = Image.open(img_path).convert('RGB')
            img_ls.append(img)
            mask_ls.append(self._load_mask(data, img))

        image_width, image_height = img_ls[0].size
        result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
        for i, img in enumerate(img_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_image.paste(img, (x, y))

        result_mask = Image.new("L", (2 * image_width, 2 * image_height))
        for i, img_mask in enumerate(mask_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_mask.paste(img_mask, (x, y))

        return result_image, result_mask

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        random_number = random.random()
        if self.training and random_number < self.aug_rate:
            img, img_mask = self.combine_img(cls_name)
        else:
            img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img_mask = self._load_mask(data, img)
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}
