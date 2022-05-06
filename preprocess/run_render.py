import os
import subprocess
from tqdm import tqdm
from subprocess import DEVNULL, STDOUT, check_call

processes = set()
max_processes = 8
categories = ['03001627', '04379243']
directories = ['../datasets/text2shape-data/ShapeNetCore.v2/03001627', '../datasets/text2shape-data/ShapeNetCore.v2/04379243']
for idx, directory in enumerate(directories):
    subdirs = os.listdir(directory)
    for subdir in tqdm(subdirs):
        path = directory + '/' + subdir + '/models'
        model_path = path + '/model_normalized.obj'
        save_path = '../datasets/224/' + categories[idx] + '/' + subdir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        processes.add(subprocess.Popen(["Blender", "--background", "--python", "render_blender.py", "--", "--output_folder", save_path, "--obj", model_path])) #要先把Blender下载好后，放入path，这样才能用Blender命令in terminal

        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])


processes = set()
max_processes = 8
v1_dir = '../datasets/ShapeNetCore.v1/04379243'
v1_sub_dirs = os.listdir(v1_dir)
v2_dir = '../datasets/ShapeNetCore.v2/04379243'
v2_sub_dirs = os.listdir(v2_dir)
difference_dirs = []
for sub_dir in v1_sub_dirs:
    if not sub_dir in v2_sub_dirs:
        difference_dirs.append(sub_dir)
difference_dirs = [v1_dir + '/' + difference_dir for difference_dir in difference_dirs]

subdirs = difference_dirs
for subdir in tqdm(subdirs):
    path = subdir
    model_path = path + '/model.obj'
    save_path = '../datasets/224/04379243/' + subdir.split('/')[-1]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    processes.add(subprocess.Popen(["Blender", "--background", "--python", "render_blender.py", "--", "--output_folder", save_path, "--obj", model_path]))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
