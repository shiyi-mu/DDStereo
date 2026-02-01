# BlockMatching algorithm
import cv2
import os
import numpy as np
from tqdm import tqdm
import skimage.measure


def save_disp_image(imageSets_txt, ori_id, result_dir):
    sub_dir = os.path.dirname(os.path.dirname(imageSets_txt))
    ori_image2_file = os.path.join(sub_dir, "image_2", ori_id+".png")
    ori_image3_file = os.path.join(sub_dir, "image_3", ori_id+".png")
    left_image = cv2.imread(ori_image2_file)
    right_image = cv2.imread(ori_image3_file)
    gray_image1 = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    matcher = cv2.StereoBM_create(192, 25)
    # stereo_matcher = cv2.StereoBM_create(192, 25)
    disparity_left = matcher.compute(gray_image1, gray_image2)
    disparity_left[disparity_left < 0] = 0
    disparity_left = disparity_left.astype(np.uint16)
    disparity_left = skimage.measure.block_reduce(disparity_left, (4,4), np.max)
    # print(">>>>mushiy>>>>", result_dir)
    cv2.imwrite(os.path.join(result_dir, "P2"+ori_id+".png"), disparity_left)


def get_disp(dataset_predix, dataset_id_txt):
    imageSets_txt = os.path.join(dataset_predix, dataset_id_txt)
    sub_dir = os.path.dirname(os.path.dirname(imageSets_txt))
    result_dir = os.path.join(sub_dir, "cv2disps")
    os.makedirs(result_dir, exist_ok=True)

    with open(imageSets_txt, "r") as f:
        lines = f.readlines()
    # 内层循环
    # inner_loop = tqdm(lines, desc='in subset :{}'.format(imageSets_txt))
    print(">>>>mushiy>>>> len(lines) :", len(lines))
    for line in tqdm(lines):
        line = line.strip()
        ori_id = line
        save_disp_image(imageSets_txt, ori_id, result_dir)

if __name__ == '__main__':
    dataset_predix = "/data3/Argoverse/argo-in-kitti-format/training"
    dataset_id_txt = "/data3/Argoverse/argo-in-kitti-format/training/ImageSets/train.txt"
    get_disp(dataset_predix, dataset_id_txt)
    
