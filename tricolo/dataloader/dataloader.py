import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from tricolo.dataloader.dataset import ClrDataset

def collate_fn(batch):
    default_collate_items = ['model_id', 'category', 'text', 'tokens', 'images']

    locs = []
    feats = []
    data = []
    for i, item in enumerate(batch):
        _locs = batch[i]['voxels']['locs']
        locs.append(torch.cat([_locs, torch.LongTensor(_locs.shape[0],1).fill_(i)],1))
        feats.append(batch[i]['voxels']['feats'])

        data.append({k:item[k] for k in default_collate_items})

    locs = torch.cat(locs)
    feats = torch.cat(feats)
    data = default_collate(data)
    data['voxels'] = {'locs': locs, 'feats': feats}
    return data

class ClrDataLoader(object):
    def __init__(self, dset, batch_size, sparse_model, num_workers, train_json_file, val_json_file, test_json_file, partnet_anno_path, image_size, voxel_size, use_voxel_color=True, root_npz_file='./datasets/all_npz/', root_partnet_file='./datasets/partnet/chair_hier'):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_json_file = train_json_file
        self.val_json_file = val_json_file
        self.test_json_file = test_json_file
        self.partnet_anno_path = partnet_anno_path
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.use_voxel_color = use_voxel_color
        self.sparse_model = sparse_model
        self.root_npz_file = root_npz_file
        self.root_partnet_file = root_partnet_file
        self.dset = dset
    
    def collate_feats(self, b):
        model_id_list = []
        parnet_anno_id_list = []
        category_list = []
        test_list = []
        tokens_list = []
        images_list = []
        voxels_list = []
        struct_tree_list = []
        point_list = []
        edge_list = []
        graph_size_list = []
        labels_list = []
        labels_num_list = []
        labels_one_hot_list = []
        for model_id, parnet_anno_id, category, text, tokens, images, voxels, struct_tree, graph in b:
            model_id_list.append(model_id)
            parnet_anno_id_list.append(parnet_anno_id)
            category_list.append(category)
            test_list.append(text)
            tokens_list.append(tokens)
            images_list.append(images)
            voxels_list.append(voxels)
            struct_tree_list.append(struct_tree)
            point_list.append(graph['points'])
            edge_list.append(graph['edges'])
            graph_size_list.append(graph['N'])
            labels_list.append(graph['labels'])
            labels_num_list.append(graph['labels_num'])
            labels_one_hot_list.append(graph['labels_one_hot'])

        tokens_array = torch.from_numpy(np.stack(tokens_list, axis=0))
        images_array = torch.from_numpy(np.stack(images_list, axis=0))
        voxels_array = torch.from_numpy(np.stack(voxels_list, axis=0))
        points = torch.cat(point_list, dim=0)
        labels_one_hot = torch.cat(labels_one_hot_list, dim=0)
        graph_size = torch.LongTensor(graph_size_list)
        cum_graph_size = [graph_size[:i].sum() for i in range(len(graph_size_list)+1)]
        edges = torch.cat([edge_list[i] + cum_graph_size[i] for i in range(len(graph_size_list))], dim=0)

        data_dict = {'model_id': model_id_list,
                    'parnet_anno_id': parnet_anno_id_list,
                    'category': category_list,
                    'text': test_list,
                    'tokens': tokens_array,
                    'images': images_array,
                    'voxels': voxels_array,
                    'struct_tree': struct_tree_list,
                    'points': points,
                    'graph_size': graph_size,
                    'edges': edges,
                    'labels': labels_list,
                    'labels_num': labels_num_list,
                    'labels_one_hot': labels_one_hot
        }
        return data_dict

    def get_data_loaders(self):
        if self.dset == 'shapenet':
            print('Using Shapenet Dataset')
            train_dataset = ClrDataset(anno_path=self.partnet_anno_path, json_file=self.train_json_file, sparse_model=self.sparse_model, image_size=self.image_size, voxel_size=self.voxel_size, use_voxel_color=self.use_voxel_color, root_npz_file=self.root_npz_file, root_partnet_file=self.root_partnet_file)
            valid_dataset = ClrDataset(anno_path=self.partnet_anno_path, json_file=self.val_json_file, sparse_model=self.sparse_model, image_size=self.image_size, voxel_size=self.voxel_size, use_voxel_color=self.use_voxel_color,root_npz_file=self.root_npz_file, root_partnet_file=self.root_partnet_file)
            test_dataset = ClrDataset(anno_path=self.partnet_anno_path, json_file=self.test_json_file, sparse_model=self.sparse_model, image_size=self.image_size, voxel_size=self.voxel_size, use_voxel_color=self.use_voxel_color,root_npz_file=self.root_npz_file, root_partnet_file=self.root_partnet_file)
        else:
            raise('Implement Other Dataset')

        if self.sparse_model:
            train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
            valid_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
            test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True, collate_fn=self.collate_feats)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True, collate_fn=self.collate_feats)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=True, collate_fn=self.collate_feats)

        print('Training file: {}, Size: {}'.format(self.train_json_file, len(train_loader.dataset)))
        print('Val file: {}, Size: {}'.format(self.val_json_file, len(valid_loader.dataset)))
        print('Test file: {}, Size: {}'.format(self.test_json_file, len(test_loader.dataset)))

        return train_loader, valid_loader, test_loader
