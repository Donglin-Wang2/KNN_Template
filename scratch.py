import os
from utils import *

import torch
import torchmetrics
from PIL import Image
from tabulate import tabulate
import open3d as o3d
import matplotlib.pyplot as plt

def test_iou():
    pred = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,3]).long()
    target = torch.tensor([0,0,1,2,0,1,1,2,0,1,2,1]).long()
    print(torchmetrics.functional.jaccard_index(pred, target, 3, average='macro'))

def test_custom_iou():
    pred = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,3]).long()
    target = torch.tensor([0,0,1,2,0,1,1,2,0,1,2,1]).long()
    haha = iou(pred, target)
    print(haha)

def test_num_part():
    data_root = './data/shapenet_part/processed/'
    points = read_pkl(os.path.join(data_root, 'points_by_cat.pkl'))
    point_labs = read_pkl(os.path.join(data_root, 'point_labels_by_cat.pkl'))
    temp_idx = read_pkl(os.path.join(data_root, 'temp_index.pkl'))

    cat_num_part = {cat:len(np.unique(point_labs[cat][temp_idx[cat][0]])) for cat in temp_idx.keys()}

    print(cat_num_part)
    print(len(temp_idx['Lamp']))

def test_write_obj():
    obj = {1:2, 3:4}
    write_pkl(obj, './logs/test.pkl')
    pass

def test_write_vis():
    data_root = './data/shapenet_part/processed/'
    points = read_pkl(os.path.join(data_root, 'points_by_cat.pkl'))['Mug'][0]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    vis = o3d.visualization.Visualizer()
    vis.add_geometry(pcd)
    image = vis.capture_screen_float_buffer()
    image = np.array(image)
    print(image, image.shape)
    im = Image.fromarray(np.asarray(image))
    im.save("./logs/filename.jpeg")
    pass

def test_plt():
    a = [1,2,3,4,5]
    plt.plot(a)
    plt.show()

# test_custom_iou()
# test_iou()
# test_num_part()

# test_write_obj()
# test_write_vis()
# test_plt()


# a = {"test": ["mug_8012f52dd0a4d2f718a93a45bf780820_0.185757357645.json", "mug_b811555ccf5ef6c4948fa2daa427fe1f_0.186238808762.json", "mug_6faf1f04bde838e477f883dde7397db2_0.175368861409.json", "mug_d38295b8d83e8cdec712af445786fe_0.170523817582.json", "mug_34ae0b61b0d8aaf2d7b20fded0142d7a_0.176904532147.json", "mug_46955fddcc83a50f79b586547e543694_0.167572957647.json", "mug_6c379385bf0a23ffdec712af445786fe_0.170523817582.json", "mug_387b695db51190d3be276203d0b1a33f_0.182910105153.json"], "train": ["mug_e94e46bc5833f2f5e57b873e4f3ef3a4_0.174047229816.json", "mug_f626192a5930d6c712f0124e8fa3930b_0.179616522537.json", "mug_1be6b2c84cdab826c043c2d07bb83fc8_0.180525504625.json", "mug_3093367916fb5216823323ed0e090a6f_0.170406127513.json", "mug_e9499e4a9f632725d6e865157050a80e_0.176069247205.json", "mug_8f6c86feaa74698d5c91ee20ade72edc_0.181324765872.json", "mug_83827973c79ca7631c9ec1e03e401f54_0.184434917227.json", "mug_10f6e09036350e92b3f21f1137c3c347_0.191847137181.json", "mug_4b8b10d03552e0891898dfa8eb8eefff_0.188155605535.json", "mug_633379db14d4d2b287dd60af81c93a3c_0.187570927943.json", "mug_cf777e14ca2c7a19b4aad3cc5ce7ee8_0.175163566197.json", "mug_3d3e993f7baa4d7ef1ff24a8b1564a36_0.174196301867.json", "mug_c34718bd10e378186c6c61abcbd83e5a_0.167391094174.json", "mug_5fe74baba21bba7ca4eec1b19b3a18f8_0.176847688635.json", "mug_9af98540f45411467246665d3d3724c_0.176784113203.json", "mug_1305b9266d38eb4d9f818dd0aa1a251_0.19174781093.json", "mug_61c10dccfa8e508e2d66cbf6a91063_0.170698949971.json", "mug_9d8c711750a73b06ad1d789f3b2120d0_0.173353283405.json", "mug_a637500654ca8d16c97cfc3e8a6b1d16_0.164812066612.json", "mug_fad118b32085f3f2c2c72e575af174cd_0.187040581981.json", "mug_b6f30c63c946c286cf6897d8875cfd5e_0.171770636885.json", "mug_ea127b5b9ba0696967699ff4ba91a25_0.172410328284.json", "mug_ff1a44e1c1785d618bca309f2c51966a_0.185506966051.json", "mug_586e67c53f181dc22adf8abaa25e0215_0.180489122776.json", "mug_162201dfe14b73f0281365259d1cf342_0.177424511788.json", "mug_3143a4accdc23349cac584186c95ce9b_0.173381842834.json", "mug_128ecbc10df5b05d96eaf1340564a4de_0.186054580816.json", "mug_1f035aa5fc6da0983ecac81e09b15ea9_0.179434953483.json", "mug_d75af64aa166c24eacbe2257d0988c9c_0.192071914133.json", "mug_85a2511c375b5b32f72755048bac3f96_0.17295003916.json", "mug_48e260a614c0fd4434a8988fdcee4fde_0.18251173138.json", "mug_46ed9dad0440c043d33646b0990bb4a_0.170298279529.json", "mug_b7e705de46ebdcc14af54ba5738cb1c5_0.180524420078.json", "mug_f7d776fd68b126f23b67070c4a034f08_0.171770636885.json", "mug_336122c3105440d193e42e2720468bf0_0.172826484955.json", "mug_71995893d717598c9de7b195ccfa970_0.174004328013.json", "mug_17952a204c0a9f526c69dceb67157a66_0.18730917851.json", "mug_1bc5d303ff4d6e7e1113901b72a68e7c_0.177077338009.json", "mug_d7ba704184d424dfd56d9106430c3fe_0.179175612694.json"]}
# print(len(a["train"]))

n_k_result = read_pkl('./logs/knn_n_K.pkl')
print(tabulate(n_k_result, tablefmt="pipe"))