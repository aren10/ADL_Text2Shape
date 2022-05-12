import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import argparse
import os

def imscatter(xs, ys, images, ax=None, zoom=1):
    
    # valid_xs = []
    # valid_ys = []
    # for i in range(len(images)):
    #     # print(images[i])
    #     if os.path.exists(images[i]):
    #         print("exists!")
    images = [plt.imread(image) for image in images]
    masked_images = []
    for image in images:
        # alpha_channel = np.ones((image.shape[0], image.shape[1], 1))
        alpha_channel = (np.min(image, axis=-1, keepdims=True) != 1.).astype(float)
        # alpha_channel[image_background] = 0.
        masked_images.append(np.concatenate([image, alpha_channel], axis=-1))
    images = masked_images
    print(images[0].shape)
    print("Loaded images.")
    images = [OffsetImage(image, zoom=zoom) for image in images]

    # x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image0 in zip(xs, ys, images):
        ab = AnnotationBbox(image0, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([xs, ys]))
    ax.autoscale()
    return artists

def run_tsne(embedding, image_paths, title, save=None):

    tsne = TSNE(n_components=2, metric="cosine", init='pca')


    print("Fitting TSNE...")
    reduced_data = tsne.fit_transform(embedding)
    print("Done fitting.")

    plt.figure(figsize=(18, 12))
    ax = plt.subplot(111)
    ax.set_title(title)

    n_points = 150
    indices = np.arange(reduced_data.shape[0])
    np.random.shuffle(indices)
    indices = indices[:n_points]
    reduced_data = reduced_data[indices]
    new_image_paths = []
    for i in range(len(indices)):
        new_image_paths.append(image_paths[indices[i]])
    image_paths = new_image_paths

    imscatter(reduced_data[:n_points,0], reduced_data[:n_points,1], image_paths[:n_points], ax=ax, zoom=0.20)
    if save is not None:
        plt.savefig(save)
    # plt.show()

def show_tsne_results(shape_rep, data_split, save=None):
    data_dir = None
    title = ""
    if shape_rep == "structure":
        data_dir = "Shapenet_S" 
        title = "StructureNet Model - "
    if shape_rep == "voxels":
        data_dir = "Shapenet_V_wocolor"
        title = "Voxel Model - "
    if shape_rep == "voxels_colored":
        data_dir = "Shapenet_V_embed" 
        title = "Colored Voxel Model - "
    if shape_rep == "cross_attention":
        data_dir = "ShapeNet_G_CrossAttention"
        title = "Cross Attention Model - "
    if data_dir is None:
        print(f"'{shape_rep}' is not a valid model option. Please try again.")
        exit(1)
    pkl_file = os.path.join(data_dir, f"{data_split}.pkl")
    data = np.load(pkl_file, allow_pickle=True)['ModelID_Fshape_Ftext_Text_PartnetID']

    shape_embedding = np.stack([t[1] for t in data] ,axis=0).astype(float).squeeze()
    text_embedding = np.stack([t[2] for t in data] ,axis=0).astype(float).squeeze()
    # image_paths = [os.path.join("1_view_shade", f"{t[0]}.png") for t in data]
    if shape_rep in ["structure", "voxels_colored", "cross_attention"]:
        image_paths = [os.path.join("1_view_texture_keyshot_1sec_alpha", f"{t[0]}.png") for t in data]
    elif shape_rep in ["voxels"]: #voxels have list of ids for some reason
        image_paths = [os.path.join("1_view_texture_keyshot_1sec_alpha", f"{t[0][0]}.png") for t in data]

    shape_embedding /= np.linalg.norm(shape_embedding, axis=-1, keepdims=True)
    text_embedding /= np.linalg.norm(text_embedding, axis=-1, keepdims=True)

    save_path = os.path.join("visualizations", data_split, shape_rep+"_" + data_split + "_shape_embeddings.png")
    run_tsne(shape_embedding, image_paths, title + "Shape Embeddings", save=save_path)
    save_path = os.path.join("visualizations", data_split, shape_rep+"_" + data_split + "_text_embeddings.png")
    run_tsne(text_embedding, image_paths, title + "Text Embeddings", save=save_path)

    # print("Fitting TSNE...")
    # reduced_data = tsne.fit_transform(shape_embedding)
    # print("Done fitting.")

    # ax = plt.subplot(111)

    # n_points = 100
    # indices = np.arange(reduced_data.shape[0])
    # np.random.shuffle(indices)
    # indices = indices[:n_points]
    # reduced_data = reduced_data[indices]
    # new_image_paths = []
    # for i in range(len(indices)):
    #     new_image_paths.append(image_paths[indices[i]])
    # image_paths = new_image_paths

    # imscatter(reduced_data[:n_points,0], reduced_data[:n_points,1], image_paths[:n_points], ax=ax, zoom=0.1)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-rep", type=str, default="structure", help="'structure' | 'voxels' | 'voxels_colored' | 'cross_attention'")
    parser.add_argument("--data-split", type=str, default="valid", help="Data split to use - 'train' | 'valid' | 'test'")
    args = parser.parse_args()

    assert(args.shape_rep in ['structure', 'voxels', 'voxels_colored', 'cross_attention'])
    assert(args.data_split in ['train', 'valid', 'test'])

    show_tsne_results(args.shape_rep, args.data_split)