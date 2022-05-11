import os
import json
import jsonlines
import cv2 as cv
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

from .tree import Tree
from collections import namedtuple

class ClrDataset(Dataset):
    def __init__(self, anno_path, json_file, sparse_model, image_size, voxel_size, use_voxel_color, root_npz_file='./datasets/all_npz/', root_partnet_file='./datasets/partnet/chair_hier'):
        shapenet_clr_frame = []
        with jsonlines.open(json_file) as reader:
            for obj in reader:
                shapenet_clr_frame.append(obj)
        shape2part_anno_path = os.path.join(anno_path, 'text2shape-data/shapenet/')
        morethan10_anno_path = os.path.join(anno_path, 'partnet/chair_geo/')
        valid_partnet_anno_ids = []
        split = ['train', 'test', 'val']
        for s in split:
            valid_part_net_anno_path = os.path.join(morethan10_anno_path, s + '_no_other_less_than_10_parts.txt')
            with open(valid_part_net_anno_path, 'r') as fp:
                lines = fp.readlines()
                lines = [line.rstrip() for line in lines]
            valid_partnet_anno_ids += lines

        part2shape_dict = {}
        split = ['train', 'test', 'val']
        for s in split:
            part_net_anno_path = os.path.join(shape2part_anno_path, 'partnet_chair.' + s +'.json')
            with open(part_net_anno_path, 'r') as fp:
                data = json.load(fp)
            for m in data:
                if m['model_id'] not in part2shape_dict:
                    part2shape_dict[m['model_id']] = m['anno_id']
        
        self.clr_frame = []
        for obj in shapenet_clr_frame:
            if obj['model'] in part2shape_dict and part2shape_dict[obj['model']] in valid_partnet_anno_ids:
                obj['parnet_anno_id'] = part2shape_dict[obj['model']]
                self.clr_frame.append(obj)
        self.clr_frame = self.clr_frame
        self.root_npz_file = root_npz_file
        self.root_partnet_file = root_partnet_file
      
        print('Image Resolution: {}, Voxel Resolution: {}'.format(image_size, voxel_size))
        self.image_size = image_size #见config
        self.voxel_size = voxel_size
        self.use_voxel_color = use_voxel_color
        self.sparse_model = sparse_model
        print("# of training text-shape pairs is:", len(self.clr_frame))

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        model_id = self.clr_frame[idx]['model']
        category = self.clr_frame[idx]['category']
        parnet_anno_id = self.clr_frame[idx]['parnet_anno_id']
        
        #############################################################
        # For Debugging Purpose. We don't need voxel modality so far.
        # voxels = np.zeros((4, self.voxel_size, self.voxel_size, self.voxel_size))
        path = self.root_npz_file + category + '/' + model_id + '.npz'
        data = np.load(path)
        if self.voxel_size == 32:
            voxel_data = data['voxel32']
        elif self.voxel_size == 64:
            voxel_data = data['voxel64']
        elif self.voxel_size == 128:
            voxel_data = data['voxel128']
        else:
            raise('Not supported voxel size')
        coords, colors = voxel_data
        coords = coords.astype(int)
        if self.use_voxel_color:
            voxels = np.zeros((4, self.voxel_size, self.voxel_size, self.voxel_size))
        else:
            voxels = np.zeros((1, self.voxel_size, self.voxel_size, self.voxel_size))

        for i in range(coords.shape[0]):
            if self.use_voxel_color:
                voxels[:3, coords[i, 0], coords[i, 1], coords[i, 2]] = colors[i]
                voxels[-1, coords[i, 0], coords[i, 1], coords[i, 2]] = 1
            else:
                voxels[0, coords[i, 0], coords[i, 1], coords[i, 2]] = 1

        #############################################################
        # StructureNet Data: Structure + Geometry
        #obj = self.load_object(os.path.join(self.root_partnet_file, parnet_anno_id+'.json'), load_geo=True)  
        obj = None
        #graph = self.load_graph
        graph = self.load_graph(os.path.join(self.root_partnet_file, parnet_anno_id+'.json'))
        #############################################################
        # For Debugging Purpose. We don't need image modality so far.
        images = np.zeros((12, 3, self.image_size, self.image_size))
        # images = data['images']
        # if self.image_size != 224:
        #     resized = []
        #     for i in range(images.shape[0]):
        #         image = images[i].transpose(1, 2, 0)
        #         image = cv.resize(image, dsize=(self.image_size, self.image_size))
        #         resized.append(image)
        #     resized = np.array(resized)
        #     images = resized.transpose(0, 3, 1, 2)
            
        
        text = self.clr_frame[idx]['caption']
        text = text.replace("\n", "")

        tokens = np.asarray(self.clr_frame[idx]['arrays'])

        if self.sparse_model:
            grid = np.transpose(voxels, (1, 2, 3, 0))
            grid = grid / 255.
            a, b = [], []
            a = np.array(grid[:, :, :, -1].nonzero()).transpose((1, 0))
            b = grid[a[:, 0], a[:, 1], a[:, 2], :3]
            a = torch.from_numpy(np.array(a)).long()
            b = torch.from_numpy(np.array(b)).float()
            locs = a
            feats = b

            data_dict = {'model_id': model_id,
                        'parnet_anno_id': parnet_anno_id, 
                        'category': category,
                        'text': text,
                        'tokens': tokens,
                        'images': images.astype(np.float32),
                        'voxels': {'locs': locs, 'feats': feats},
                        'struct_tree': {'locs': locs, 'feats': feats}}
            return data_dict
        else:
            images = images.astype(np.float32)
            voxels = voxels.astype(np.float32)
            return model_id, parnet_anno_id, category, text, tokens, images, voxels, obj, graph

    def get_partnet_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id+'.json'), \
                load_geo=self.load_geo)
        return obj

    @staticmethod
    def load_graph(fn):
        chair_part_cats = ['arm_connector', 'arm_holistic_frame', 'arm_horizontal_bar', 'arm_near_vertical_bar', 'arm_sofa_style', 'arm_writing_table', 'back_connector', 'back_frame', 'back_frame_horizontal_bar', 'back_frame_vertical_bar', 'back_holistic_frame', 'back_single_surface', 'back_support', 'back_surface', 'back_surface_horizontal_bar', 'back_surface_vertical_bar', 'bar_stretcher', 'caster', 'caster_stem', 'central_support', 'chair_arm', 'chair_back', 'chair_base', 'chair_head', 'chair_seat', 'foot', 'foot_base', 'footrest', 'head_connector', 'headrest', 'knob', 'leg', 'lever', 'mechanical_control', 'other', 'pedestal', 'pedestal_base', 'regular_leg_base', 'rocker', 'runner', 'seat_frame', 'seat_frame_bar', 'seat_holistic_frame', 'seat_single_surface', 'seat_support', 'seat_surface', 'seat_surface_bar', 'star_leg_base', 'star_leg_set', 'wheel']

        geo_fn = fn.replace('_hier', '_geo').replace('json', 'npz')
        geo_data = np.load(geo_fn)
        all_points = torch.from_numpy(geo_data['parts'])
        flatten_fn = fn.replace('_hier', '_flatten').replace('json', 'pkl')
        with open(flatten_fn, 'rb') as f:
            nodes, edges = pickle.load(f)
        points = []
        labels = []
        labels_one_hot = []
        labels_num = []
        for node in nodes:
            points.append(all_points[node['id']])
            label = node['label']
            labels.append(label)
            label_num = chair_part_cats.index(label)
            labels_num.append(label_num)
            label_one_hot = torch.zeros(len(chair_part_cats))
            label_one_hot[label_num] = 1
            labels_one_hot.append(label_one_hot)
        points = torch.stack(points)
        labels_one_hot = torch.stack(labels_one_hot)
        assert points.shape[0] == len(nodes)
        N = points.shape[0]

        edges = torch.LongTensor(edges)
        data = {
            'points': points,
            'edges': edges,
            'N': N,
            'labels': labels,
            'labels_num': labels_num,
            'labels_one_hot': labels_one_hot
        }
        return data

    @staticmethod
    def load_object(fn, load_geo=False):
        if load_geo:
            geo_fn = fn.replace('_hier', '_geo').replace('json', 'npz')
            geo_data = np.load(geo_fn)

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            node = Tree.Node(
                part_id=node_json['id'],
                is_leaf=('children' not in node_json),
                label=node_json['label'])

            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32).view(1, -1, 3)

            if load_geo:
                node.geo = torch.tensor(geo_data['parts'][node_json['id']], dtype=torch.float32).view(1, -1, 3)

            if 'box' in node_json:
                node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        return obj

    @staticmethod
    def save_object(obj, fn):

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=obj.root, parent_json=None, parent_child_idx=None)]

        obj_json = None

        # traverse the tree, converting child nodes of each node to json
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node

            node_json = {
                'id': node.part_id,
                'label': f'{node.label if node.label is not None else ""}'}

            if node.geo is not None:
                node_json['geo'] = node.geo.cpu().numpy().reshape(-1).tolist()

            if node.box is not None:
                node_json['box'] = node.box.cpu().numpy().reshape(-1).tolist()

            if len(node.children) > 0:
                node_json['children'] = []
            for child in node.children:
                node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=node_json, parent_child_idx=len(node_json['children'])-1))

            if len(node.edges) > 0:
                node_json['edges'] = []
            for edge in node.edges:
                node_json['edges'].append(edge)
                if 'params' in edge:
                    node_json['edges'][-1]['params'] = node_json['edges'][-1]['params'].cpu().numpy().reshape(-1).tolist()

            if parent_json is None:
                obj_json = node_json
            else:
                parent_json['children'][parent_child_idx] = node_json

        with open(fn, 'w') as f:
            json.dump(obj_json, f)
