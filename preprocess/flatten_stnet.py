import json
import pickle
from pathlib import Path
from pyquaternion import Quaternion
import numpy as np
import torch

category = "chair"
save_dir = f"../datasets/partnet/{category}_flatten"
data_dir = f"../datasets/partnet/{category}_hier"
part_data_dir = "/data_hdd/data_v0"
from tqdm import tqdm

class Part():
    def __init__(self):
        pass

def get_box_quat(box):
    center = box[:3]
    size = box[3:6]
    xdir = box[6:9]
    xdir /= np.linalg.norm(xdir)
    ydir = box[9:]
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)
    zdir /= np.linalg.norm(zdir)
    rotmat = np.vstack([xdir, ydir, zdir]).T
    q = Quaternion(matrix=rotmat)
    quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
    box_quat = np.hstack([center, size, quat]).astype(np.float32)
    return box_quat

def box_from_box_quat(box_quat):
    center = box_quat[:3]
    size = box_quat[3:6]
    q = Quaternion(box_quat[6], box_quat[7], box_quat[8], box_quat[9])
    rotmat = q.rotation_matrix
    box = np.hstack([center, size, rotmat[:, 0].flatten(), rotmat[:, 1].flatten()]).astype(np.float32)
    return box

def get_box_representation2(box):
    center = box[:3]
    size = box[3:6]
    xdir = box[6:9]
    xdir /= np.linalg.norm(xdir)
    ydir = box[9:]
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)
    zdir /= np.linalg.norm(zdir)
    rotmat = np.vstack([xdir, ydir, zdir]).T.flatten()
    box_rep = np.hstack([center, size, rotmat]).astype(np.float32)
    #print(rotmat.flatten().reshape(3,3))
    return box_rep

def box_from_representation2(box_rep):
    center = box_rep[:3]
    size = box_rep[3:6]
    rotmat = box_rep[6:].reshape(3,3)
    box = np.hstack([center, size, rotmat[:, 0].flatten(), rotmat[:, 1].flatten()]).astype(np.float32)
    return box


def getCuboid(a):
    c = Cuboid("c")
    device = 'cpu'
    c.dims = torch.tensor([
        2 * a['xd'],
        2 * a['yd'],
        2 * a['zd'],
    ], dtype = torch.float).to(device)
    c.pos = torch.tensor(a['center'], dtype = torch.float).to(device)
    c.rfnorm = torch.tensor(a['xdir'], dtype = torch.float).to(device)
    c.tfnorm = torch.tensor(a['ydir'], dtype = torch.float).to(device)
    c.ffnorm = torch.tensor(a['zdir'], dtype = torch.float).to(device)

    return c

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f

def getObj(cuboids):

    verts = torch.tensor([],dtype=torch.float)
    faces = torch.tensor([],dtype=torch.long)
    
    for cube in cuboids:
        v, f = cube.getTris()
        if v is not None and f is not None:
            faces = torch.cat((faces, (f + verts.shape[0])))
            verts = torch.cat((verts, v))
    
    return verts, faces

def writeObj(verts, faces, outfile):
    faces = faces.clone()
    faces += 1
    with open(outfile, 'w') as f:
        for a, b, c in verts.tolist():
            f.write(f'v {a} {b} {c}\n')

        for a, b, c in faces.tolist():
            f.write(f"f {a} {b} {c}\n")

def jsonToProps(json):
    json = np.array(json)
    center = np.array(json[:3])

    xd = json[3] / 2
    yd = json[4] / 2
    zd = json[5] / 2
    xdir = json[6:9]
    xdir /= np.linalg.norm(xdir)
    ydir = json[9:]
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)
    zdir /= np.linalg.norm(zdir)
    return {
        'center': center,
        'xd': xd,
        'yd': yd,
        'zd': zd,
        'xdir': xdir,
        'ydir': ydir,
        'zdir': zdir
    }


def render_point(points, labels=None,probs=None):
    render = np.zeros((256,256))
    for j in range(points.shape[0]):
        if labels is not None:
            if labels[j] == 0:
                continue
        p = points[j]
        x,y,z = p
        a = -y + z * 0.5
        b = x - z * 0.5

        a = int((a+1) * 128)
        b = int((b+1) * 128)
        if probs is None:
            coeff = 1
        else:
            coeff = float(probs[j]) ** 1.5
        if 1<a<254 and 1<b<254:
            render[a-1:a+2,b-2:b+2] += 100 * coeff

    return render

def pairwise_dist(p1, p2):
    return (p1**2).sum(dim=1).view(-1,1) + (p2**2).sum(dim=1).view(1,-1) - 2 * p1@p2.t()

def get_obj(objs, partnet_shape_dir):
    all_verts = []
    all_faces = []
    offset = 0
    for obj in objs:
        #print(obj)
        verts, faces = load_obj(f"{partnet_shape_dir}/objs/{obj}.obj")

        faces -= 1
        #print(node.partnet_shape_dir)
        used_vs = set()
        for fidx in range(faces.shape[0]):
            face = faces[fidx]
            for vidx in range(3):
                used_vs.add(face[vidx])
        used_vs = sorted(list(used_vs))

        verts = verts[used_vs]

        vert_map = {used_vs[a]:a for a in range(len(used_vs))}

        for fidx in range(faces.shape[0]):
            for vidx in range(3):
                faces[fidx][vidx] = vert_map[faces[fidx][vidx]]

        #writeObj(verts, faces+1, f"{obj}.obj")
        faces += offset
        offset += verts.shape[0]

        all_verts.append(verts)
        all_faces.append(faces)


        #print(np.concatenate((verts, verts), axis=0).shape)
        #print(verts.shape)
        #quit()
    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    return verts, faces
    
if __name__ == "__main__":
    files = []
    for (i,filename) in enumerate(Path(data_dir).glob('*.json')):
        files.append(filename)
    all_cats = set()

    save_data = True


    count = 0
    with torch.no_grad():
        for i in tqdm(range(len(files))):
            chair_id = str(files[i]).split('/')[-1][:-5]
            with open(files[i], 'rb') as f:
                shape = json.load(f)

            geo_dir = str(files[i]).replace("hier", "geo").replace("json", "npz")

            try:
                with np.load(geo_dir) as f:
                    all_points = torch.from_numpy(f['parts'])
            except:
                continue

            partnet_shape_dir = f"{part_data_dir}/{chair_id}"
            with open(f"{partnet_shape_dir}/result_after_merging.json") as f:
                part_json = json.load(f)

            assert len(part_json) == 1

            partnet_part_dict = {}

            root_node = part_json[0]

            def parse_partnet_level(shape):
                if 'children' not in shape:
                    partnet_part_dict[shape['id']] = shape
                else:
                    for child in shape['children']:
                        parse_partnet_level(child)

            parse_partnet_level(root_node)

            parts = []
            cuboids = []
            
            graph_edges = []  

            def parse_one_level(shape, parent, level):
                for idx, part in enumerate(shape['children']):
                    new_part = Part()
                    new_part.parent = shape
                    new_part.id = part['id']
                    new_part.idx = idx
                    new_part.label = part['label']
                    new_part.adj = []
                    new_part.children = []
                    if parent is root:
                        new_part.parent = None
                    else:
                        new_part.parent = parent
                    parent.children.append(new_part)

                    new_part.points = all_points[new_part.id]
                    #assert new_part.points.shape[0] == 1000 and new_part.points.shape[1] == 3
                    #all_cats.add(f"{new_part.label}_of_{parent.label}_{level}")
                    all_cats.add(f"{new_part.label}")
                    assert len(part['box']) == 12
                    #box = part['box']
                    new_part.box_original = part['box']

                    box_quat = get_box_representation2(part['box'])
                    new_part.box = box_quat
                    #assert all(s >= 0 for s in new_part.size)
                    parts.append(new_part)
                    
                    if 'children' in part:
                        parse_one_level(part, new_part, level+1)

                if "edges" in shape:
                    for edge in shape['edges']:
                        if edge['type'] == 'ADJ':
                            a = edge['part_a']
                            b = edge['part_b']

                            a = parent.children[a]
                            b = parent.children[b]
                            a.adj.append(b)
                            b.adj.append(a)


            root = Part()
            root.label = shape['label']
            root.children = []
            try:
                parse_one_level(shape, root, 0)
            except:
                continue
            #continue
            num_leaf_nodes = len([p for p in parts if len(p.children) == 0])
            while not all(len(part.children) == 0 for part in parts):
                for part in parts:
                    if len(part.children) > 0:
                        break

                parts.remove(part)
                for child in part.children:
                    child.parent = None

                for adj_part in part.adj:
                    adj_part.adj.remove(part)
                    p2 = adj_part.points
                    for child in part.children:
                        p1 = child.points

                        dists = pairwise_dist(p1, p2) ** 0.5
                        assert dists.shape[0] == 1000 and dists.shape[1] == 1000
                        #print(dists.min())
                        if dists.min() < 0.025:
                            adj_part.adj.append(child)
                            child.adj.append(adj_part)
                
            assert len(parts) == num_leaf_nodes
            
            all_valid_contacts = True
            st_ids = []
            found = True
            for part in parts:
                st_ids.append(part.id)
                if part.id not in partnet_part_dict:
                    found = False
                    break
                assert part.parent is None
                assert len(part.children) == 0

                p1 = part.points

            part_to_idx = {part:i for (i, part) in enumerate(parts)}

            for part in parts:
                u = part_to_idx[part]
                for adj_part in part.adj:
                    v = part_to_idx[adj_part]
                    assert u != v
                    if u < v:
                        graph_edges.append((u,v))

            nodes = []
            for part in parts:
                #print(part.__dict__)
                del part.points
                del part.parent
                del part.adj
                del part.children
                node = part.__dict__
                nodes.append(node)
                #print(part.points.shape)
            if save_data:
            #if False:
                with open(f"{save_dir}/{chair_id}.pkl", 'wb') as f:
                    pickle.dump((nodes, graph_edges), f, pickle.HIGHEST_PROTOCOL)

        all_cats = sorted(list(all_cats))
        print(all_cats)
