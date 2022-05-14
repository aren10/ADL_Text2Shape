# Exploring Structural Features for Text to Shape Retrieval
This repo is implementation of Exploring Structural Features for Text to Shape Retrieval(a class project for CSCI2952N). This repo is inheritate from: ([*Project Page*](https://3dlg-hcvc.github.io/tricolo/)).

## Environment Installation
1. Install required packages and [CLIP](https://github.com/openai/CLIP)
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c conda-forge tensorboard
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
2. Install [sparseconvolution](https://github.com/facebookresearch/SparseConvNet)
```
git clone https://github.com/facebookresearch/SparseConvNet.git
cd SparseConvNet/
bash develop.sh
```
<!-- conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch -->

## Dataset Download
1. Create datasets folder
```
mkdir datasets
cd datasets
```
2. Download StructureNet dataset
Please follow the instruction here from [PartNet](https://partnet.cs.stanford.edu/) official webset. Note we only need the chair category data. 

Please also download the json files for train-text split from here this [repo](https://github.com/daerduoCarey/partnet_dataset/tree/master/stats/train_val_test_split).

3. Download [Text2Shape](http://text2shape.stanford.edu/) Dataset and unzip
```
wget http://text2shape.stanford.edu/dataset/text2shape-data.zip
unzip text2shape-data.zip
```

* ShapeNet

    1. Download [Colored Voxels](http://text2shape.stanford.edu/) and unzip
    ```
    cd text2shape-data/shapenet
    wget http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_32_solid.zip
    wget http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_64_solid.zip
    wget http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_128_solid.zip
    unzip nrrd_256_filter_div_32_solid.zip
    unzip nrrd_256_filter_div_64_solid.zip
    unzip nrrd_256_filter_div_128_solid.zip
    ```
    2. Download [ShapeNet](https://shapenet.org/) v1 and v2 (for multi-view), and place in datasets folder.

The directory structure should look like:
```
This Repository
|--datasets
    |--partnet 
        |--chair_geo
        |--chair_hier
    |--text2shape
        |--shapenet
            |--nrrd_256_filter_div_32_solid
            |--nrrd_256_filter_div_64_solid
            |--nrrd_256_filter_div_128_solid
            |--partnet_chair.text.json
            |--partnet_chair.train.json
            |--partnet_chair.val.json
            | Other meta files upzipped...
```

## Dataset Preprocessing
1. Save dataset into json files
```
cd preprocess
python create_modelid_caption_mapping.py --dataset shapenet
```

2. Generate npz files for each model
```
python gen_all_npz.py
```

## Train Model
1. ShapeNet Dataset

    Train the Bi-modal Voxel model on ShapeNet
    ```
    python run_retrieval.py --config_file tricolo/configs/shapenet_V.yaml
    ```
    Train the Bi-modal Voxel without color model on ShapeNet
    ```
    python run_retrieval.py --config_file tricolo/configs/shapenet_V_wocolor.yaml
    ```
    Train the Bi-modal StructureNet-based model on ShapeNet
    ```
    python run_retrieval.py --config_file tricolo/configs/shapenet_S.yaml
    ```
    Train the Bi-modal PartGNN-based model on ShapeNet
    ```
    python run_retrieval.py --config_file tricolo/configs/shapenet_F.yaml
    ```

## Evaluate Model
1. Evaluate trained model. Here enter the trained log folder name under ./logs/retrieval for the flag --exp. For the --split flag enter either 'valid' or 'test' to evaluate on that dataset split.
```
# Example
python run_retrieval_val.py --exp CONFIGNAME
```