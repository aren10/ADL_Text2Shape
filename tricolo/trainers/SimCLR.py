"""
Code modified from https://github.com/edreisMD/ConVIRT-pytorch/blob/master/train.py
"""

#python run_retrieval.py --config_file /Users/adamren/Desktop/tricolo/tricolo/configs/shapenet.yaml 用这个做train

import os
import clip
import json
import pickle
import logging
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from tricolo.loss.nt_xent import NTXentLoss
from tricolo.models.retrieval_model import ModelCLR
from tricolo.metrics.eval_retrieval import compute_cross_modal

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def _save_config_file(checkpoints_folder, config):
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
        with open(os.path.join(checkpoints_folder, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

class SimCLR(object):
    def __init__(self, dataset, config, param_id=-1):
        self.dset = config['dset']
        self.config = config
        self.device = 'cuda'
        if self.config['train']:
            log_dir = 'logs/retrieval/' + datetime.datetime.now().strftime("%b%d_%H-%M-%S")
            if param_id != -1:
                log_dir = log_dir+'-'+str(param_id)
            self.writer =  SummaryWriter(log_dir)
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        
        self.use_voxel = config['model']['use_voxel']
        self.use_struct = config['model']['use_struct']
        self.use_flatten = config['model']['use_flatten']
        self.use_voxel_color = config['dataset']['use_voxel_color']
        self.tri_modal = config['model']['tri_modal']
        self.num_images = config['model']['num_images']
        self.multiplier = 12 // self.num_images
    
    def train(self):
        train_loader, valid_loader, _ = self.dataset.get_data_loaders()

        model = ModelCLR(self.dset, self.config['dataset']['voxel_size'], self.config['sparse_model'], self.use_voxel_color, **self.config["model"]).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), eval(self.config['learning_rate']), weight_decay=eval(self.config['weight_decay']))

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder, self.config)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print(f'Training...')
        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch {epoch_counter}')
            for data_dict in tqdm(train_loader):
                torch.cuda.empty_cache()

                xls = data_dict['tokens'].to(self.device)

                voxels, images, struct_tree = None, None, None
                graph = None

                if self.tri_modal:
                    if self.config['sparse_model']:
                        voxels = {}
                        voxels['locs'] = data_dict['voxels']['locs'].to(self.device)
                        voxels['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        voxels = data_dict['voxels'].to(self.device)
                    images = data_dict['images'][:, ::self.multiplier].to(self.device)
                elif self.use_voxel:
                    if self.config['sparse_model']:
                        voxels = {}
                        voxels['locs'] = data_dict['voxels']['locs'].to(self.device)
                        voxels['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        voxels = data_dict['voxels'].to(self.device)
                elif self.use_struct:
                    if self.config['sparse_model']:
                        struct_tree = {}
                        struct_tree['locs'] = data_dict['voxels']['locs'].to(self.device)
                        struct_tree['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        struct_tree = []
                        struct_tree_ = data_dict['struct_tree']
                        for obj in struct_tree_:
                            struct_tree.append(obj.to(self.device))
                elif self.use_flatten:
                    graph = {}
                    graph['points'] = data_dict['points'].to(self.device)
                    graph['edges'] = data_dict['edges']
                    graph['N'] = data_dict['graph_size']
                    graph['labels'] = data_dict['labels']
                    graph['labels_num'] = data_dict['labels_num']
                    graph['one_hot'] = data_dict['labels_one_hot'].to(self.device)
                    voxels = data_dict['voxels'].to(self.device)
                else:
                    images = data_dict['images'][:, ::self.multiplier].to(self.device)

                optimizer.zero_grad()
                z_voxels, z_struct, z_flatten, z_images, zls = model(voxels, struct_tree, graph, images, xls)
                zls = F.normalize(zls, dim=1)
                if self.tri_modal:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    z_images = F.normalize(z_images, dim=1)
                    loss = self.nt_xent_criterion(z_images, z_voxels) + self.nt_xent_criterion(z_voxels, zls) + self.nt_xent_criterion(z_images, zls)
                elif self.use_voxel:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    loss = self.nt_xent_criterion(z_voxels, zls)
                elif self.use_struct:
                    z_struct = F.normalize(z_struct, dim=1)
                    loss = self.nt_xent_criterion(z_struct, zls)
                elif self.use_flatten:
                    z_flatten = F.normalize(z_flatten, dim=1)
                    loss = self.nt_xent_criterion(z_flatten, zls)
                else:
                    z_images = F.normalize(z_images, dim=1)
                    loss = self.nt_xent_criterion(z_images, zls)
                loss.backward()
                optimizer.step()

                # For Debug Purpose
                # cmd = 'nvidia-smi'
                # returned_value = os.system(cmd)  # returns the exit code in unix
                # print(returned_value)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                n_iter += 1
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(epoch_counter)))

                valid_loss = self._validate(model, valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

    def _load_pre_trained_weights(self, model, log_dir):
        try:
            checkpoints_folder = os.path.join(log_dir, 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))

            modify = []
            for key, value in state_dict.items():
                if key.split('.')[0] == 'bert_model':
                    new_name = key.split('.')
                    new_name[0] = 'text_model'
                    new_name = '.'.join(new_name)
                    modify.append((new_name, key))

            for item in modify:
                new_name, key = item
                state_dict[new_name] = state_dict.pop(key)

            if 'fc.weight' in state_dict:
                state_dict['text_fc.weight'] = state_dict.pop('fc.weight')
            if 'fc.bias' in state_dict:
                state_dict['text_fc.bias'] = state_dict.pop('fc.bias')

            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pre-trained model with success. Loading from {checkpoints_folder}")
        except FileNotFoundError:
            print("Pre-trained weights not found.")
            raise
        return model

    def _validate(self, model, valid_loader, n_iter):
        model.eval()
        with torch.no_grad():
            counter = 0
            valid_loss = 0.0
            print(f'Validation step')
            for data_dict in tqdm(valid_loader):
                xls = data_dict['tokens'].to(self.device)

                voxels, images, struct_tree = None, None, None
                graph = None

                if self.tri_modal:
                    if self.config['sparse_model']:
                        voxels = {}
                        voxels['locs'] = data_dict['voxels']['locs'].to(self.device)
                        voxels['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        voxels = data_dict['voxels'].to(self.device)
                    images = data_dict['images'][:, ::self.multiplier].to(self.device)
                elif self.use_voxel:
                    if self.config['sparse_model']:
                        voxels = {}
                        voxels['locs'] = data_dict['voxels']['locs'].to(self.device)
                        voxels['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        voxels = data_dict['voxels'].to(self.device)
                elif self.use_struct:
                    if self.config['sparse_model']:
                        struct_tree = {}
                        struct_tree['locs'] = data_dict['voxels']['locs'].to(self.device)
                        struct_tree['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        struct_tree = []
                        struct_tree_ = data_dict['struct_tree']
                        for obj in struct_tree_:
                            struct_tree.append(obj.to(self.device))
                elif self.use_flatten:
                    graph = {}
                    graph['points'] = data_dict['points'].to(self.device)
                    graph['edges'] = data_dict['edges']
                    graph['N'] = data_dict['graph_size']
                    graph['labels'] = data_dict['labels']
                    graph['labels_num'] = data_dict['labels_num']
                    graph['one_hot'] = data_dict['labels_one_hot'].to(self.device)
                    voxels = data_dict['voxels'].to(self.device)
                else:
                    images = data_dict['images'][:, ::self.multiplier].to(self.device)

                #z_voxels, z_struct, z_images, zls = model(voxels, struct_tree, images, xls)
                z_voxels, z_struct, z_flatten, z_images, zls = model(voxels, struct_tree, graph, images, xls)
                zls = F.normalize(zls, dim=1)
                if self.tri_modal:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    z_images = F.normalize(z_images, dim=1)
                    loss = self.nt_xent_criterion(z_images, z_voxels) + self.nt_xent_criterion(z_voxels, zls) + self.nt_xent_criterion(z_images, zls)
                elif self.use_voxel:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    loss = self.nt_xent_criterion(z_voxels, zls)
                elif self.use_struct:
                    z_struct = F.normalize(z_struct, dim=1)
                    loss = self.nt_xent_criterion(z_struct, zls)
                elif self.use_flatten:
                    z_flatten = F.normalize(z_flatten, dim=1)
                    loss = self.nt_xent_criterion(z_flatten, zls)
                else:
                    z_images = F.normalize(z_images, dim=1)
                    loss = self.nt_xent_criterion(z_images, zls)

                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
    
    def save_output(self, log_dir, eval_loader='valid'):
        with torch.no_grad():
            train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

            # self.config["model"]["use_struct_pretrain"] = self.config["model"]["pretraining"]
            model = ModelCLR(self.dset, self.config['dataset']['voxel_size'], self.config['sparse_model'], self.use_voxel_color, **self.config["model"]).to(self.device) #model is from retrieval_model which is the training model
            model = self._load_pre_trained_weights(model, log_dir)
            model.eval()

            model_test_folder = os.path.join(log_dir, eval_loader)
            _save_config_file(model_test_folder, self.config)

            print('Testing...')

            loader = None
            if eval_loader == 'valid':
                print('Evaluating on val loader')
                loader = valid_loader
            elif eval_loader == 'test':
                print('Evaluating on test loader')
                loader = test_loader

            modelids = []
            text_embeds = []
            shape_embeds = []
            category_list = []
            all_caption_indices = []
            for data_dict in tqdm(loader):
                xls = data_dict['tokens'].to(self.device)

                voxels, images, struct_tree = None, None, None

                if self.tri_modal:
                    if self.config['sparse_model']:
                        voxels = {}
                        voxels['locs'] = data_dict['voxels']['locs'].to(self.device)
                        voxels['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        voxels = data_dict['voxels'].to(self.device)
                    images = data_dict['images'][:, ::self.multiplier].to(self.device)
                elif self.use_voxel:
                    if self.config['sparse_model']:
                        voxels = {}
                        voxels['locs'] = data_dict['voxels']['locs'].to(self.device)
                        voxels['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        voxels = data_dict['voxels'].to(self.device)
                elif self.use_struct:
                    if self.config['sparse_model']:
                        struct_tree = {}
                        struct_tree['locs'] = data_dict['voxels']['locs'].to(self.device)
                        struct_tree['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        struct_tree = []
                        struct_tree_ = data_dict['struct_tree']
                        for obj in struct_tree_:
                            struct_tree.append(obj.to(self.device))
                elif self.use_flatten:
                    graph = {}
                    graph['points'] = data_dict['points'].to(self.device)
                    graph['edges'] = data_dict['edges']
                    graph['N'] = data_dict['graph_size']
                    graph['labels'] = data_dict['labels']
                    graph['labels_num'] = data_dict['labels_num']
                    graph['one_hot'] = data_dict['labels_one_hot'].to(self.device)
                else:
                    images = data_dict['images'][:, ::self.multiplier].to(self.device)

                z_voxels, z_struct, z_images, zls = model(voxels, struct_tree, images, xls)
                zls = F.normalize(zls, dim=1)
                if self.tri_modal:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    z_images = F.normalize(z_images, dim=1)
                    # shape_embeds.append(z_images.detach().cpu().numpy())
                    # shape_embeds.append(z_voxels.detach().cpu().numpy())
                    shape_embeds.append((z_images+z_voxels).detach().cpu().numpy())
                elif self.use_voxel:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    shape_embeds.append(z_voxels.detach().cpu().numpy())
                elif self.use_struct:
                    z_struct = F.normalize(z_struct, dim=1)
                    shape_embeds.append(z_struct.detach().cpu().numpy())
                else:
                    z_images = F.normalize(z_images, dim=1)
                    shape_embeds.append(z_images.detach().cpu().numpy())

                text_embeds.append(zls.detach().cpu().numpy())
                modelids.extend(list(data_dict['model_id']))
                category_list.extend(list(data_dict['category']))
                caption_indices = data_dict['tokens'].detach().cpu().numpy()
                
                for cap in caption_indices:
                    all_caption_indices.append(cap)

            all_text = np.vstack(text_embeds)
            all_shape = np.vstack(shape_embeds)
            assert all_text.shape[0] == all_shape.shape[0]

            tuples = []
            embeddings_dict = {}
            for i in range(all_text.shape[0]):
                new_tup = (all_caption_indices[i], category_list[i], modelids[i], all_text[i], all_shape[i]) 
                tuples.append(new_tup)
            embeddings_dict['caption_embedding_tuples'] = tuples
            save_output_path = os.path.join(model_test_folder, 'output.p')
            with open(save_output_path, 'wb') as f:
                pickle.dump(embeddings_dict, f)
                print(f"saved output dict to {save_output_path}")
        return save_output_path

    def save_output_clip(self, log_dir, eval_loader='valid'):
        print('Using CLIP to eval')
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

        with torch.no_grad():
            train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

            model_test_folder = os.path.join(log_dir, eval_loader)

            print('Testing...')

            loader = None
            if eval_loader == 'valid':
                print('Evaluating on val loader')
                loader = valid_loader
            elif eval_loader == 'test':
                print('Evaluating on test loader')
                loader = test_loader

            modelids = []
            text_embeds = []
            shape_embeds = []
            category_list = []
            all_caption_indices = []
            for data_dict in tqdm(loader):
                images = data_dict['images'].to(self.device).reshape(data_dict['images'].shape[0]*12, data_dict['images'].shape[2], data_dict['images'].shape[3], data_dict['images'].shape[4])
                text = clip.tokenize(data_dict['text'], truncate=True).to(self.device)

                def _transform():
                    return Compose([
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
                preprocess = _transform()
                images = preprocess(images)

                with torch.no_grad():
                    image_features = clip_model.encode_image(images).reshape(-1, 12, 512)
                    image_features = torch.mean(image_features, 1)
                    text_features = clip_model.encode_text(text)

                    image_features = F.normalize(image_features, dim=1)
                    text_features = F.normalize(text_features, dim=1)

                xls = data_dict['tokens'].to(self.device)

                shape_embeds.append(image_features.detach().cpu().numpy())
                text_embeds.append(text_features.detach().cpu().numpy())
                modelids.extend(list(data_dict['model_id']))
                category_list.extend(list(data_dict['category']))

                caption_indices = data_dict['tokens'].detach().cpu().numpy()
                
                for cap in caption_indices:
                    all_caption_indices.append(cap)

            all_text = np.vstack(text_embeds)
            all_shape = np.vstack(shape_embeds)
            assert all_text.shape[0] == all_shape.shape[0]

            tuples = []
            embeddings_dict = {}
            for i in range(all_text.shape[0]):
                new_tup = (all_caption_indices[i], category_list[i], modelids[i], all_text[i], all_shape[i]) 
                tuples.append(new_tup)
            embeddings_dict['caption_embedding_tuples'] = tuples
            save_output_path = os.path.join(model_test_folder, 'output.p')
            with open(save_output_path, 'wb') as f:
                pickle.dump(embeddings_dict, f)
                print(f"saved output dict to {save_output_path}")
        return save_output_path

    def save_output_embed(self, log_dir, eval_loader='valid'):
        with torch.no_grad():
            
            train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

            model = ModelCLR(self.dset, self.config['dataset']['voxel_size'], self.config['sparse_model'], self.use_voxel_color, **self.config["model"]).to(self.device) #model is from retrieval_model which is the training model
            model = self._load_pre_trained_weights(model, log_dir)
            model.eval()

            model_test_folder = os.path.join(log_dir, 'embed')
            _save_config_file(model_test_folder, self.config)

            print('Testing...')

            loader = None
            if eval_loader == 'valid':
                print('Evaluating on val loader')
                loader = valid_loader
            elif eval_loader == 'test':
                print('Evaluating on test loader')
                loader = test_loader
            elif eval_loader == 'train':
                print('Evaluating on train loader')
                loader = train_loader

            embed_list = []
            for data_dict in tqdm(loader):

                xls = data_dict['tokens'].to(self.device)

                voxels, images, struct_tree, z_flatten = None, None, None, None
                data_tuple = (data_dict['model_id'], )

                if self.tri_modal:
                    if self.config['sparse_model']:
                        voxels = {}
                        voxels['locs'] = data_dict['voxels']['locs'].to(self.device)
                        voxels['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        voxels = data_dict['voxels'].to(self.device)
                    images = data_dict['images'][:, ::self.multiplier].to(self.device)
                elif self.use_voxel:
                    if self.config['sparse_model']:
                        voxels = {}
                        voxels['locs'] = data_dict['voxels']['locs'].to(self.device)
                        voxels['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        voxels = data_dict['voxels'].to(self.device)
                elif self.use_struct:
                    if self.config['sparse_model']:
                        struct_tree = {}
                        struct_tree['locs'] = data_dict['voxels']['locs'].to(self.device)
                        struct_tree['feats'] = data_dict['voxels']['feats'].to(self.device)
                    else:
                        struct_tree = []
                        struct_tree_ = data_dict['struct_tree']
                        for obj in struct_tree_:
                            struct_tree.append(obj.to(self.device))
                elif self.use_flatten:
                    graph = {}
                    graph['points'] = data_dict['points'].to(self.device)
                    graph['edges'] = data_dict['edges']
                    graph['N'] = data_dict['graph_size']
                    graph['labels'] = data_dict['labels']
                    graph['labels_num'] = data_dict['labels_num']
                    graph['one_hot'] = data_dict['labels_one_hot'].to(self.device)
                else:
                    images = data_dict['images'][:, ::self.multiplier].to(self.device)

                z_voxels, z_struct, z_flatten, z_images, zls = model(voxels, struct_tree, graph, images, xls)
                zls = F.normalize(zls, dim=1)
                
                if self.tri_modal:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    z_images = F.normalize(z_images, dim=1)
                    shape_feature = ((z_images+z_voxels).detach().cpu().numpy())
                    # shape_embeds.append(z_images.detach().cpu().numpy())
                    # shape_embeds.append(z_voxels.detach().cpu().numpy())
                    # shape_embeds.append((z_images+z_voxels).detach().cpu().numpy())
                elif self.use_voxel:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    shape_feature = (z_voxels.detach().cpu().numpy())
                elif self.use_struct:
                    z_struct = F.normalize(z_struct, dim=1)
                    shape_feature = (z_struct.detach().cpu().numpy())
                elif self.use_flatten:
                    z_flatten = F.normalize(z_flatten, dim=1)
                    shape_feature = (z_flatten.detach().cpu().numpy())
                else:
                    z_images = F.normalize(z_images, dim=1)
                    shape_feature = (z_images.detach().cpu().numpy())

                for i in range(shape_feature.shape[0]):
                    data_tuple = (data_dict['model_id'][i], shape_feature[i], zls.detach().cpu().numpy()[i], data_dict['text'][i], data_dict['parnet_anno_id'][i])
                    embed_list.append(data_tuple)

            embeddings_dict = {}
            embeddings_dict['ModelID_Fshape_Ftext_Text_PartnetID'] = embed_list
            save_output_path = os.path.join(model_test_folder, eval_loader + '.pkl')
            with open(save_output_path, 'wb') as f:
                pickle.dump(embeddings_dict, f)
                print(f"saved output dict to {save_output_path}")
        return save_output_path

    def test(self, log_dir, clip=False, eval_loader='valid'):
        model_test_folder = os.path.join(log_dir, eval_loader)
        if not os.path.exists(model_test_folder):
            os.makedirs(model_test_folder)

        if clip:
            embeddings_path = self.save_output_clip(log_dir, eval_loader)
        else:
            if os.path.isfile(os.path.join(model_test_folder, 'output.p')):
                embeddings_path = os.path.join(model_test_folder, 'output.p')
            else:
                embeddings_path = self.save_output(log_dir, eval_loader)
        
        metric = 'cosine'
        dset = self.config['dset']

        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)

        render_dir = os.path.join(os.path.dirname(embeddings_path), 'nearest_neighbor_renderings')
        pr_at_k = compute_cross_modal(dset, embeddings_dict, model_test_folder, metric, concise=render_dir)
        return pr_at_k

    def inference(self, log_dir, clip=False, eval_loader='valid'):
        model_test_folder = os.path.join(log_dir, 'embed')
        if not os.path.exists(model_test_folder):
            os.makedirs(model_test_folder)

        if clip:
            embeddings_path = self.save_output_clip(log_dir, eval_loader)
        else:
            if os.path.isfile(os.path.join(model_test_folder, eval_loader + '.pkl')):
                embeddings_path = os.path.join(model_test_folder, eval_loader + '.pkl')
            else:
                embeddings_path = self.save_output_embed(log_dir, eval_loader)
        
        print('Model saved at:', embeddings_path)
