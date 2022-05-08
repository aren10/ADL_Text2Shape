"""
    This file defines the point cloud shape VAE/AE model.
"""

import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch import nn
import torch_scatter

from .compute_sym import *
from .utils import load_checkpoint
from tricolo.dataloader.tree import Tree


class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        encode = torch.relu(self.mlp1(x))
        mu = self.mlp2mu(encode)

        if self.probabilistic:
            logvar = self.mlp2var(encode)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class PartFeatSampler(nn.Module):

    def __init__(self, feature_size, probabilistic=True):
        super(PartFeatSampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp2mu = nn.Linear(feature_size, feature_size)
        self.mlp2var = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        mu = self.mlp2mu(x)

        if self.probabilistic:
            logvar = self.mlp2var(x)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class PartEncoder(nn.Module):

    def __init__(self, feat_len):
        super(PartEncoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_len, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feat_len)

        self.sampler = PartFeatSampler(feature_size=feat_len, probabilistic=False)

    def forward(self, pc):
        net = pc.transpose(2, 1)
        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=2)[0]
        net = self.sampler(net)

        return net


class NodeEncoder(nn.Module):

    def __init__(self, geo_feat_len, node_feat_len):
        super(NodeEncoder, self).__init__()

        self.part_encoder = PartEncoder(geo_feat_len)

        self.mlp1 = nn.Linear(4, geo_feat_len)
        self.mlp2 = nn.Linear(2 * geo_feat_len, node_feat_len)

    def forward(self, geo):
        num_point = geo.size(1)

        center = geo.mean(dim=1)
        geo = geo - center.unsqueeze(dim=1).repeat(1, num_point, 1)
        scale = geo.pow(2).sum(dim=2).max(dim=1)[0].sqrt().view(-1, 1)
        box_feat = torch.cat([center, scale], dim=1)
        box_feat = torch.relu(self.mlp1(box_feat))

        geo = geo / scale.unsqueeze(dim=1).repeat(1, num_point, 3)
        geo_feat = self.part_encoder(geo)

        all_feat = torch.cat([box_feat, geo_feat], dim=1)
        all_feat = torch.relu(self.mlp2(all_feat))

        return all_feat, geo_feat


class GNNChildEncoder(nn.Module):

    def __init__(self, node_feat_size, hidden_size, node_symmetric_type, \
            edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNChildEncoder, self).__init__()

        self.node_symmetric_type = node_symmetric_type
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num

        self.child_op = nn.Linear(node_feat_size + Tree.num_sem, hidden_size)
        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size*2+edge_type_num, hidden_size))

        self.parent_op = nn.Linear(hidden_size*(self.num_iterations+1), node_feat_size)

    """
        Input Arguments:
            child feats: b x max_childs x feat_dim
            child exists: b x max_childs x 1
            edge_type_onehot: b x num_edges x edge_type_num
            edge_indices: b x num_edges x 2
    """
    def forward(self, child_feats, child_exists, edge_type_onehot, edge_indices):
        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        num_edges = edge_indices.shape[1]

        if batch_size != 1:
            raise ValueError('Currently only a single batch is supported.')

        # perform MLP for child features
        child_feats = torch.relu(self.child_op(child_feats))
        hidden_size = child_feats.size(-1)

        # zero out non-existent children
        child_feats = child_feats * child_exists
        child_feats = child_feats.view(1, max_childs, -1)

        # combine node features before and after message-passing into one parent feature
        iter_parent_feats = []
        if self.node_symmetric_type == 'max':
            iter_parent_feats.append(child_feats.max(dim=1)[0])
        elif self.node_symmetric_type == 'sum':
            iter_parent_feats.append(child_feats.sum(dim=1))
        elif self.node_symmetric_type == 'avg':
            iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
        else:
            raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

        if self.num_iterations > 0 and num_edges > 0:
            edge_feats = edge_type_onehot

        edge_indices_from = edge_indices[:, :, 0].view(-1, 1).expand(-1, hidden_size)

        # perform Graph Neural Network for message-passing among sibling nodes
        for i in range(self.num_iterations):
            if num_edges > 0:
                # MLP for edge features concatenated with adjacent node features
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[0, :, 0], :], # start node features
                    child_feats[0:1, edge_indices[0, :, 1], :], # end node features
                    edge_feats], dim=2) # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))
                node_edge_feats = node_edge_feats.view(num_edges, -1)

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, hidden_size)

            # combine node features before and after message-passing into one parent feature
            if self.node_symmetric_type == 'max':
                iter_parent_feats.append(child_feats.max(dim=1)[0])
            elif self.node_symmetric_type == 'sum':
                iter_parent_feats.append(child_feats.sum(dim=1))
            elif self.node_symmetric_type == 'avg':
                iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
            else:
                raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

        # concatenation of the parent features from all iterations (as in GIN, like skip connections)
        parent_feat = torch.cat(iter_parent_feats, dim=1)

        # back to standard feature space size
        parent_feat = torch.relu(self.parent_op(parent_feat))

        return parent_feat


class RecursiveEncoder(nn.Module):

    def __init__(self, config=None, variational=True, probabilistic=False):
        super(RecursiveEncoder, self).__init__()
        if config is None:
            config = torch.load(os.path.join(os.path.dirname(__file__), 'pc_ae_chair', 'conf.pth'))
        self.conf = config

        Tree.load_category_info(self.conf.category)

        self.node_encoder = NodeEncoder(geo_feat_len=config.geo_feat_size, node_feat_len=config.feature_size)

        self.child_encoder = GNNChildEncoder(
            node_feat_size=config.feature_size,
            hidden_size=config.hidden_size,
            node_symmetric_type=config.node_symmetric_type,
            edge_symmetric_type=config.edge_symmetric_type,
            num_iterations=config.num_gnn_iterations,
            edge_type_num=len(config.edge_types))

        if variational:
            self.sample_encoder = Sampler(feature_size=config.feature_size, \
                    hidden_size=config.hidden_size, probabilistic=probabilistic)
        
        load_checkpoint(
            models=[self],
            model_names=['encoder'],
            dirname=os.path.join(os.path.dirname(__file__), 'pc_ae_chair'),
            epoch=226,
            strict=True)

    def encode_node(self, node):
        all_feat, geo_feat = self.node_encoder(node.geo)
        node.geo_feat = geo_feat

        if node.is_leaf:
            return all_feat
        else:
            # get features of all children
            child_feats = []
            for child in node.children:
                cur_child_feat = torch.cat([self.encode_node(child), child.get_semantic_one_hot()], dim=1)
                child_feats.append(cur_child_feat.unsqueeze(dim=1))
            child_feats = torch.cat(child_feats, dim=1)

            if child_feats.shape[1] > self.conf.max_child_num:
                raise ValueError('Node has too many children.')

            # pad with zeros
            if child_feats.shape[1] < self.conf.max_child_num:
                padding = child_feats.new_zeros(child_feats.shape[0], \
                        self.conf.max_child_num-child_feats.shape[1], child_feats.shape[2])
                child_feats = torch.cat([child_feats, padding], dim=1)

            # 1 if the child exists, 0 if it is padded
            child_exists = child_feats.new_zeros(child_feats.shape[0], self.conf.max_child_num, 1)
            child_exists[:, :len(node.children), :] = 1
            # get feature of current node (parent of the children)
            edge_type_onehot, edge_indices = node.edge_tensors(
                edge_types=self.conf.edge_types, device=child_feats.device, type_onehot=True)

            return self.child_encoder(child_feats, child_exists, edge_type_onehot, edge_indices)

    def encode_structure(self, obj):
        root_latent = self.encode_node(obj.root)
        return self.sample_encoder(root_latent)


class LeafClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 1)

    def forward(self, input_feature):
        output = torch.relu(self.mlp1(input_feature))
        output = self.mlp2(output)
        return output


class SampleDecoder(nn.Module):
    
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, input_feature):
        output = torch.relu(self.mlp1(input_feature))
        output = torch.relu(self.mlp2(output))
        return output


class PartDecoder(nn.Module):

    def __init__(self, feat_len, num_point):
        super(PartDecoder, self).__init__()
        self.num_point = num_point

        self.mlp1 = nn.Linear(feat_len, feat_len)
        self.mlp2 = nn.Linear(feat_len, feat_len)
        self.mlp3 = nn.Linear(feat_len, num_point*3)

        self.bn1 = nn.BatchNorm1d(feat_len)
        self.bn2 = nn.BatchNorm1d(feat_len)

    def forward(self, net):
        net = torch.relu(self.bn1(self.mlp1(net)))
        net = torch.relu(self.bn2(self.mlp2(net)))
        net = self.mlp3(net).view(-1, self.num_point, 3)

        return net


class NodeDecoder(nn.Module):

    def __init__(self, geo_feat_len, node_feat_len, num_point):
        super(NodeDecoder, self).__init__()

        self.mlp1 = nn.Linear(node_feat_len, 3)
        self.mlp2 = nn.Linear(node_feat_len, 1)
        self.mlp3 = nn.Linear(node_feat_len, geo_feat_len)

        self.part_decoder = PartDecoder(geo_feat_len, num_point)

    def forward(self, net):
        geo_center = torch.tanh(self.mlp1(net))
        geo_scale = torch.sigmoid(self.mlp2(net))

        geo_feat = self.mlp3(net)
        geo_local = self.part_decoder(geo_feat)

        return geo_local, geo_center, geo_scale, geo_feat


class GNNChildDecoder(nn.Module):

    def __init__(self, node_feat_size, hidden_size, max_child_num, \
            edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNChildDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.hidden_size = hidden_size
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num

        self.mlp_parent = nn.Linear(node_feat_size, hidden_size*max_child_num)
        self.mlp_exists = nn.Linear(hidden_size, 1)
        self.mlp_sem = nn.Linear(hidden_size, Tree.num_sem)
        self.mlp_child = nn.Linear(hidden_size, node_feat_size)
        self.mlp_edge_latent = nn.Linear(hidden_size*2, hidden_size)

        self.mlp_edge_exists = nn.ModuleList()
        for i in range(edge_type_num):
            self.mlp_edge_exists.append(nn.Linear(hidden_size, 1))

        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size*3+edge_type_num, hidden_size))

        self.mlp_child = nn.Linear(hidden_size*(self.num_iterations+1), hidden_size)
        self.mlp_sem = nn.Linear(hidden_size, Tree.num_sem)
        self.mlp_child2 = nn.Linear(hidden_size, node_feat_size)

    def forward(self, parent_feature):
        batch_size = parent_feature.shape[0]
        feat_size = parent_feature.shape[1]

        if batch_size != 1:
            raise ValueError('Only batch size 1 supported for now.')

        parent_feature = torch.relu(self.mlp_parent(parent_feature))
        child_feats = parent_feature.view(batch_size, self.max_child_num, self.hidden_size)

        # node existence
        child_exists_logits = self.mlp_exists(child_feats.view(batch_size*self.max_child_num, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)

        # edge features
        edge_latents = torch.cat([
            child_feats.view(batch_size, self.max_child_num, 1, feat_size).expand(-1, -1, self.max_child_num, -1),
            child_feats.view(batch_size, 1, self.max_child_num, feat_size).expand(-1, self.max_child_num, -1, -1)
            ], dim=3)
        edge_latents = torch.relu(self.mlp_edge_latent(edge_latents))

        # edge existence prediction
        edge_exists_logits_per_type = []
        for i in range(self.edge_type_num):
            edge_exists_logits_cur_type = self.mlp_edge_exists[i](edge_latents).view(\
                    batch_size, self.max_child_num, self.max_child_num, 1)
            edge_exists_logits_per_type.append(edge_exists_logits_cur_type)
        edge_exists_logits = torch.cat(edge_exists_logits_per_type, dim=3)

        """
            decoding stage message passing
            there are several possible versions, this is a simple one:
            use a fixed set of edges, consisting of existing edges connecting existing nodes
            this set of edges does not change during iterations
            iteratively update the child latent features
            then use these child latent features to compute child features and semantics
        """
        # get edges that exist between nodes that exist
        edge_indices = torch.nonzero(edge_exists_logits > 0)
        edge_types = edge_indices[:, 3]
        edge_indices = edge_indices[:, 1:3]
        nodes_exist_mask = (child_exists_logits[0, edge_indices[:, 0], 0] > 0) \
                & (child_exists_logits[0, edge_indices[:, 1], 0] > 0)
        edge_indices = edge_indices[nodes_exist_mask, :]
        edge_types = edge_types[nodes_exist_mask]

        # get latent features for the edges
        edge_feats_mp = edge_latents[0:1, edge_indices[:, 0], edge_indices[:, 1], :]

        # append edge type to edge features, so the network has information which
        # of the possibly multiple edges between two nodes it is working with
        edge_type_logit = edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], :]
        edge_type_logit = edge_feats_mp.new_zeros(edge_feats_mp.shape[:2]+(self.edge_type_num,))
        edge_type_logit[0:1, range(edge_type_logit.shape[1]), edge_types] = \
                edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], edge_types]
        edge_feats_mp = torch.cat([edge_feats_mp, edge_type_logit], dim=2)

        num_edges = edge_indices.shape[0]
        max_childs = child_feats.shape[1]

        iter_child_feats = [child_feats] # zeroth iteration

        if self.num_iterations > 0 and num_edges > 0:
            edge_indices_from = edge_indices[:, 0].view(-1, 1).expand(-1, self.hidden_size)

        for i in range(self.num_iterations):
            if num_edges > 0:
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[:, 0], :], # start node features
                    child_feats[0:1, edge_indices[:, 1], :], # end node features
                    edge_feats_mp], dim=2) # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, self.hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, self.hidden_size)

            # save child features of this iteration
            iter_child_feats.append(child_feats)

        # concatenation of the child features from all iterations (as in GIN, like skip connections)
        child_feats = torch.cat(iter_child_feats, dim=2)

        # transform concatenation back to original feature space size
        child_feats = child_feats.view(-1, self.hidden_size*(self.num_iterations+1))
        child_feats = torch.relu(self.mlp_child(child_feats))
        child_feats = child_feats.view(batch_size, self.max_child_num, self.hidden_size)

        # node semantics
        child_sem_logits = self.mlp_sem(child_feats.view(-1, self.hidden_size))
        child_sem_logits = child_sem_logits.view(batch_size, self.max_child_num, Tree.num_sem)

        # node features
        child_feats = self.mlp_child2(child_feats.view(-1, self.hidden_size))
        child_feats = child_feats.view(batch_size, self.max_child_num, feat_size)
        child_feats = torch.relu(child_feats)

        return child_feats, child_sem_logits, child_exists_logits, edge_exists_logits