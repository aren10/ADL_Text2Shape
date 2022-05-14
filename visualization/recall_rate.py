import pickle
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os


# def show_img_color(array):
#     try:
#         img = Image.fromarray(array.astype('uint8'))
#     except:
#         img = array
#     img.show()


# eval_dir = 'Shapenet_S'
# eval_dir = 'Shapenet_V_wocolor'
eval_dir = 'Shapenet_V_embed'


render_dir = '1_view_texture_keyshot_1sec_alpha'
out_dir = f'vis/{eval_dir}'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

split = 'test'

# num_rows = 4
# num_cols = 5
# img_size = 224

with open(f"{eval_dir}/{split}.pkl", 'rb') as f:
    data = pickle.load(f)

for key, value in data.items():
    pass

#ModelID_Fshape_Ftext_Text_PartnetID
data = value

all_part_embeddings = [torch.from_numpy(d[1]) for d in data]

all_part_embeddings = torch.stack(all_part_embeddings)
early_version = (len(all_part_embeddings.shape) == 3)
if early_version:
    all_part_embeddings = all_part_embeddings.squeeze(1)

n = 15

total_objs = 0
in_top_n = 0

with torch.no_grad():
    for text_index, shape in enumerate(data):
        modelid, _, text_emb, text, partnetid = shape
        text_emb = torch.from_numpy(text_emb)
        if early_version:
            text_emb = text_emb.squeeze(0)
            text = text[0]
            modelid = modelid[0]
        similarity = F.cosine_similarity(text_emb.unsqueeze(0), all_part_embeddings)

        similarity = list(similarity)
        similarity = sorted([(i, float(similarity[i])) for i in range(len(similarity))], key=lambda x:-x[1])

        top_n = [x[0] for x in similarity[:n]]
        total_objs += 1
        if text_index in top_n:
            in_top_n += 1
        # print(f"Top 5 for {text_index}")
        # print([x[0] for x in similarity[:5]])
        # print('\n\n\n')
        
        # vis_img = np.zeros((img_size*(num_rows+1), img_size*num_cols, 3))

        # try:
        #     gt_img = np.array(Image.open(f'{render_dir}/{modelid}.png'))
        #     vis_img[:img_size, :img_size,:] = gt_img
        # except:
        #     pass

        # rank = 0
        # existing = []
        # for i in range(num_rows):
        #     for j in range(num_cols):
        #         target_id = data[similarity[rank][0]][0]
        #         if early_version:
        #             target_id = target_id[0]
        #         while target_id in existing:
        #             rank += 1
        #             #target_id = data[similarity[rank][0]][0]
        #             target_id = data[similarity[rank][0]][0]
        #             if early_version:
        #                 target_id = target_id[0]
        #         existing.append(target_id)
        #         try:
        #             ret_img = np.array(Image.open(f'{render_dir}/{target_id}.png'))
        #             vis_img[img_size*(1+i):img_size*(2+i), img_size*j:img_size*(j+1),:] = ret_img
        #         except:
        #             pass

        # vis_img = Image.fromarray(vis_img.astype('uint8'))
        # draw = ImageDraw.Draw(vis_img)
        # font = ImageFont.truetype("arial.ttf", 20)

        # draw.text((img_size*1.2, img_size/2), text, (255,255,255), font=font)
        # vis_img.save(f'{out_dir}/{modelid}_{text.replace(" ", "_")[0:10]}.png')
        #quit()


print(f"Top {n} RR for {eval_dir}")

print(f"Random Recall - {n/total_objs*100:.2f}")
print(f"Model Recall  - {in_top_n/total_objs*100:.2f}")