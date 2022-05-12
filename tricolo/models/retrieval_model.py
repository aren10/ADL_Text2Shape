"""
Code modified from: https://github.com/edreisMD/ConVIRT-pytorch/blob/master/models/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricolo.models.cross_attention import MAX_SEMANTIC_LABEL, MAX_PART_NUM, CrossAttention
from tricolo.models.models import cnn_encoder, cnn_encoder32, cnn_encoder_sparse, SVCNN, MVCNN
from tricolo.models.structurenet_models.model_pc import RecursiveEncoder
from tricolo.models.gnn import FlattenModel


class ModelCLR(nn.Module):
    def __init__(self, dset, voxel_size, sparse_model, use_voxel_color, out_dim, use_voxel, use_struct, use_struct_pretrain, tri_modal, num_images, image_cnn, pretraining, vocab_size, use_flatten=False, use_cross_attention=False):
        super(ModelCLR, self).__init__()

        self.dset = dset
        self.ef_dim = 32
        self.z_dim = 512
        self.out_dim = out_dim
        self.cnn_name = image_cnn
        self.use_voxel = use_voxel
        self.use_voxel_color = use_voxel_color
        self.use_struct = use_struct
        self.use_struct_pretrain = use_struct_pretrain
        self.use_flatten = use_flatten
        self.use_cross_attention = use_cross_attention
        self.tri_modal = tri_modal
        self.voxel_size = voxel_size
        self.num_images = num_images
        self.pretraining = pretraining
        self.sparse_model = sparse_model
        self.chair_part_cats = ['arm_connector', 'arm_holistic_frame', 'arm_horizontal_bar', 'arm_near_vertical_bar', 'arm_sofa_style', 'arm_writing_table', 'back_connector', 'back_frame', 'back_frame_horizontal_bar', 'back_frame_vertical_bar', 'back_holistic_frame', 'back_single_surface', 'back_support', 'back_surface', 'back_surface_horizontal_bar', 'back_surface_vertical_bar', 'bar_stretcher', 'caster', 'caster_stem', 'central_support', 'chair_arm', 'chair_back', 'chair_base', 'chair_head', 'chair_seat', 'foot', 'foot_base', 'footrest', 'head_connector', 'headrest', 'knob', 'leg', 'lever', 'mechanical_control', 'other', 'pedestal', 'pedestal_base', 'regular_leg_base', 'rocker', 'runner', 'seat_frame', 'seat_frame_bar', 'seat_holistic_frame', 'seat_single_surface', 'seat_support', 'seat_surface', 'seat_surface_bar', 'star_leg_base', 'star_leg_set', 'wheel']
        self.text_model, self.text_fc = self._get_text_encoder()
        self.embedding_layer = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.voxel_model, self.voxel_fc, \
            self.struct_model, self.struct_fc, \
                self.image_model, self.image_fc, \
                    self.flatten_model, self.flatten_fc, self.flatten_project, self.flatten_attn, self.flatten_text_proj, self.flatten_flat_proj = self._get_res_encoder()
        self.flatten_model.cats = self.chair_part_cats

    def _get_text_encoder(self):
        print("Text feature extractor: BiGRU")
        text_model = nn.GRU(input_size=256, hidden_size=128, num_layers=1, bidirectional=True)
        text_fc = nn.Linear(256, self.out_dim)
        return text_model, text_fc

    def _get_res_encoder(self):
        voxel_model = None
        voxel_fc = None
        image_model = None
        image_fc = None
        struct_model = None
        struct_fc = None
        flatten_model = None
        flatten_fc = None
        flatten_project = None
        flatten_attn = None
        flatten_flat_proj= None
        flatten_text_proj= None

        if self.dset == 'shapenet':
            if self.tri_modal:
                print('Training Tri-Modal Model')
                if self.sparse_model:
                    voxel_model = cnn_encoder_sparse(self.voxel_size, self.ef_dim, self.z_dim)
                else:
                    voxel_model = cnn_encoder(self.use_voxel_color, self.voxel_size, self.ef_dim, self.z_dim)
                voxel_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))

                svcnn = SVCNN(self.z_dim, pretraining=self.pretraining, cnn_name=self.cnn_name)
                image_model = MVCNN(self.z_dim, svcnn, cnn_name=self.cnn_name, num_views=self.num_images)
                image_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
            elif self.use_voxel:
                print('Training Bi-Modal Model')
                if self.sparse_model:
                    voxel_model = cnn_encoder_sparse(self.voxel_size, self.ef_dim, self.z_dim)
                else:
                    voxel_model = cnn_encoder(self.use_voxel_color, self.voxel_size, self.ef_dim, self.z_dim)
                voxel_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
            elif self.use_struct:
                print('Training Bi-Modal Model - StructureNet')
                if self.sparse_model:
                    struct_model = cnn_encoder_sparse(self.voxel_size, self.ef_dim, self.z_dim)
                else:
                    struct_model = RecursiveEncoder(pretrain=self.use_struct_pretrain)
                struct_fc = nn.Sequential(nn.Linear(struct_model.conf.feature_size, self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
            elif self.use_flatten:
                print('Training Bi-Modal Model - Flattened GNN')
                if self.sparse_model:
                    raise NotImplementedError
                else:
                    flatten_model = FlattenModel()
                    #voxel_model = cnn_encoder(self.use_voxel_color, self.voxel_size, self.ef_dim, self.z_dim)
                flatten_fc = nn.Sequential(nn.Linear(128+self.z_dim, self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
                if self.use_cross_attention:
                    flatten_attn = CrossAttention(256)
                    flatten_project = nn.Sequential(nn.Linear(64, 256),nn.ReLU(),nn.Linear(256,256))
                    flatten_fc = nn.Sequential(nn.Linear(256, self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
                    flatten_text_proj = nn.AvgPool1d(96)
                    flatten_flat_proj = nn.AvgPool1d(MAX_PART_NUM)
                else:
                    flatten_fc = nn.Sequential(nn.Linear(128, self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
            else:
                print('Training Bi-Modal Model')
                svcnn = SVCNN(self.z_dim, pretraining=self.pretraining, cnn_name=self.cnn_name)
                image_model = MVCNN(self.z_dim, svcnn, cnn_name=self.cnn_name, num_views=self.num_images)
                image_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
        else:
            raise('Implement Other Dataset')
        return voxel_model, voxel_fc, struct_model, struct_fc, image_model, image_fc, flatten_model, flatten_fc, flatten_project, flatten_attn, flatten_text_proj, flatten_flat_proj

    def voxel_encoder(self, xis):
        h = self.voxel_model(xis)
        h.squeeze()
        x = self.voxel_fc(h)
        return x

    def struct_encoder(self, xis):
        h_list = []
        for obj in xis:
            h = self.struct_model.encode_structure(obj)
            h_list.append(h)

        h_batch = torch.cat(h_list, dim=0)
        x = self.struct_fc(h_batch)
        return x

    def flatten_encoder2(self, xis, voxels):
        h, _ = self.flatten_model(xis, get_graph_embeddings=True, get_node_embeddings=False)
        h1 = self.voxel_model(voxels)
        h1.squeeze()
        x = self.flatten_fc(torch.cat([h, h1], dim=1))
        return x

    def flatten_encoder(self, xis, yis=None):
        h, h_list = self.flatten_model(xis, get_graph_embeddings=True, get_node_embeddings=True)
        if yis is not None:
            h_array = torch.zeros((len(h_list), MAX_PART_NUM, 64)).to(h_list[0].device)
            for i in range(len(h_list)):
                h_array[i, :h_list[i].shape[0]] = h_list[i]
            h = self.flatten_project(h_array) #[B, Max, 256]
            yis = yis.transpose(0, 1)
            h = torch.cat((h, yis), dim=1) #[B, Max+L, 256]
            h = self.flatten_attn(h).transpose(1, 2)
            h = self.flatten_flat_proj(h).squeeze()
            x = self.flatten_fc(h)
            yis = yis.transpose(1, 2)
            h = self.flatten_text_proj(yis).squeeze()
            y = torch.tanh(self.text_fc(h))
            return x, y
        else:
            x = self.flatten_fc(h)
            return x

    def image_encoder(self, xis):
        h = self.image_model(xis)
        h.squeeze()
        x = self.image_fc(h)
        return x

    def text_encoder(self, encoded_inputs):
        embed_inputs = self.embedding_layer(encoded_inputs)
        embed_inputs = torch.transpose(embed_inputs, 0, 1)

        N = embed_inputs.shape[1]

        h0 = torch.zeros(2, N, 128).cuda()
        # h0 = torch.zeros(2, N, 128)
        output, hidden = self.text_model(embed_inputs, h0)
        
        if self.use_cross_attention:
            return output
        else:
            out_emb = torch.tanh(self.text_fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
            return out_emb

    def forward(self, voxels, struct_tree, graph, images, encoded_inputs):
        z_voxels = None
        z_images = None
        z_struct = None
        z_flatten = None

        zls = self.text_encoder(encoded_inputs)

        if self.tri_modal:
            images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
            z_voxels = self.voxel_encoder(voxels)
            z_images = self.image_encoder(images)
        elif self.use_voxel:
            z_voxels = self.voxel_encoder(voxels)
        elif self.use_struct:
            z_struct = self.struct_encoder(struct_tree)
        elif self.use_flatten:
            #z_flatten = self.flatten_encoder(graph, voxels)
            if self.use_cross_attention:
                z_flatten, zls = self.flatten_encoder(graph, zls)
            else:
                z_flatten = self.flatten_encoder(graph)
        else:
            images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
            z_images = self.image_encoder(images)

        
        return z_voxels, z_struct, z_flatten, z_images, zls
