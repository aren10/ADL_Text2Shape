import os
import time
import nrrd
import pickle
import jsonlines
import cv2 as cv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def save_npz(obj): 
    obj_first = obj[0]
    voxel64_root_dir = obj[1]
    category = obj_first.split('/')[0]
    model_id = obj_first.split('/')[1]
    save_dir = '../datasets/all_npz' + '/' + category
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    voxel32 = []
    voxel128 = []

    ######################################################################################################
    '''
    voxel32
    
    voxel32, _ = nrrd.read(voxel32_root_dir + '/' + model_id + '/' + model_id + '.nrrd')
    voxel32 = voxel32.astype(np.float32)
    voxel32 = voxel32 / 255.

    mask = (voxel32[-1] > 0.9).astype(int)
    index = np.nonzero(mask)

    x = index[0].reshape(-1, 1)
    y = index[1].reshape(-1, 1)
    z = index[2].reshape(-1, 1)
    coords = np.concatenate((x, y, z), axis=1)

    r = voxel32[0][index].reshape(-1, 1)
    g = voxel32[1][index].reshape(-1, 1)
    b = voxel32[2][index].reshape(-1, 1)
    colors = np.concatenate((r, g, b), axis=1)
    voxel32 = [coords, colors]
    '''
    ######################################################################################################

    ######################################################################################################
    '''
    voxel64
    '''
    voxel64, _ = nrrd.read(voxel64_root_dir + '/' + model_id + '/' + model_id + '.nrrd')
    voxel64 = voxel64.astype(np.float32)
    voxel64 = voxel64 / 255.

    mask = (voxel64[-1] > 0.9).astype(int)
    index = np.nonzero(mask)

    x = index[0].reshape(-1, 1)
    y = index[1].reshape(-1, 1)
    z = index[2].reshape(-1, 1)
    coords = np.concatenate((x, y, z), axis=1)

    r = voxel64[0][index].reshape(-1, 1)
    g = voxel64[1][index].reshape(-1, 1)
    b = voxel64[2][index].reshape(-1, 1)
    colors = np.concatenate((r, g, b), axis=1)
    voxel64 = [coords, colors]
    ######################################################################################################

    ######################################################################################################
    '''
    voxel128
    
    voxel128, _ = nrrd.read(voxel128_root_dir + '/' + model_id + '/' + model_id + '.nrrd')
    voxel128 = voxel128.astype(np.float32)
    voxel128 = voxel128 / 255.

    mask = (voxel128[-1] > 0.9).astype(int)
    index = np.nonzero(mask)

    x = index[0].reshape(-1, 1)
    y = index[1].reshape(-1, 1)
    z = index[2].reshape(-1, 1)
    coords = np.concatenate((x, y, z), axis=1)

    r = voxel128[0][index].reshape(-1, 1)
    g = voxel128[1][index].reshape(-1, 1)
    b = voxel128[2][index].reshape(-1, 1)
    colors = np.concatenate((r, g, b), axis=1)
    voxel128 = [coords, colors]
    '''
    ######################################################################################################
    
    # view_ids = ['{}.png'.format(i) for i in range(0, 12)]
    # pre_path = '../datasets/224/{}/{}/'.format(category, model_id)
    # img_paths = [pre_path+view_id for view_id in view_ids]
    # images = []
    # for img_path in img_paths:
    #     img = np.array(cv.imread(img_path)) #c5c4e6110fbbf5d3d83578ca09f86027 cannot be rendered into differnent views' png
    #     if img is None:
    #         img = np.zeros((224, 224, 3))
    #         print("_____________________0______________________")
    #     normalized_img = img.astype(np.float32) / 255.
    #     images.append(normalized_img)
    
    # images = np.array(images).transpose(0, 3, 1, 2)
    
    np.savez_compressed(save_dir + '/' + model_id + '.npz', voxel32=voxel32, voxel64=voxel64, voxel128=voxel128, images=voxel128)

if __name__== '__main__':
    if not os.path.exists('../datasets/all_npz'):
        os.makedirs('../datasets/all_npz')

    train_json_file =   '../datasets/text2shape-data/shapenet/train_map.jsonl'
    val_json_file =     '../datasets/text2shape-data/shapenet/val_map.jsonl'
    test_json_file =    '../datasets/text2shape-data/shapenet/test_map.jsonl'
    voxel32_root_dir =  '../datasets/text2shape-data/shapenet/nrrd_256_filter_div_32_solid'
    voxel64_root_dir = '../datasets/text2shape-data/shapenet/nrrd_256_filter_div_64_solid'
    #voxel64_root_dir =  '../datasets/text2shape-data/shapenet/nrrd_256_filter_div_64_solid'
    voxel128_root_dir = '../datasets/text2shape-data/shapenet/nrrd_256_filter_div_128_solid'

    train_dataset = []
    with jsonlines.open(train_json_file) as reader:
        for obj in reader:
            train_dataset.append(obj)

    val_dataset = []
    with jsonlines.open(val_json_file) as reader:
        for obj in reader:
            val_dataset.append(obj)

    test_dataset = []
    with jsonlines.open(test_json_file) as reader:
        for obj in reader:
            test_dataset.append(obj)

    print('Train has {} samples.'.format(len(train_dataset)))
    print('Val has {} samples.'.format(len(val_dataset)))
    print('Test has {} samples.'.format(len(test_dataset)))


    all_train_obj = []
    for item in tqdm(train_dataset):
        category = item['category']
        model_id = item['model']
        path = category + '/' + model_id
        all_train_obj.append(path)
    print(len(all_train_obj))
    print(len(set(all_train_obj)))
    all_train_obj = list(set(all_train_obj))

    all_valid_obj = []
    for item in tqdm(val_dataset):
        category = item['category']
        model_id = item['model']
        path = category + '/' + model_id
        all_valid_obj.append(path)
    print(len(all_valid_obj))
    print(len(set(all_valid_obj)))
    all_valid_obj = list(set(all_valid_obj))

    all_test_obj = []
    for item in tqdm(test_dataset):
        category = item['category']
        model_id = item['model']
        path = category + '/' + model_id
        all_test_obj.append(path)
    print(len(all_test_obj))
    print(len(set(all_test_obj)))
    all_test_obj = list(set(all_test_obj))
    
    all_train_obj_input = []
    for train_obj in all_train_obj:
        all_train_obj_input.append((train_obj, voxel64_root_dir))

    all_valid_obj_input = []
    for valid_obj in all_valid_obj:
        all_valid_obj_input.append((valid_obj, voxel64_root_dir))
    
    all_test_obj_input = []
    for test_obj in all_test_obj:
        all_test_obj_input.append((test_obj, voxel64_root_dir))

    with Pool(14) as p:
        r = list(tqdm(p.imap(save_npz, all_train_obj_input))) #?????????????????????tuples????????????????????????????????????tuple
        # r = list(tqdm(p.imap(save_npz, (all_train_obj, voxel64_root_dir_list), total=len(all_train_obj))))
        r = list(tqdm(p.imap(save_npz, all_valid_obj_input)))
        r = list(tqdm(p.imap(save_npz, all_test_obj_input)))
