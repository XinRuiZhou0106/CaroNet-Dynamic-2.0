import os
import cv2
import torch
import glob
import numpy as np

# crop vessel region + zero-padding + resize
def crop_resize(img, extract_v_box):
    bbox_list = [int(extract_v_box[1]), int(extract_v_box[0]), int(extract_v_box[3]), int(extract_v_box[2])] # y1, x1, y2, x2
    img = img[bbox_list[0]:bbox_list[2],bbox_list[1]:bbox_list[3]]
    h,w = img.shape[0], img.shape[1]
    if h>w:
        padding_left, padding_right = int((h-w)/2), h-w-int((h-w)/2)
        img = cv2.copyMakeBorder(img,0,0,padding_left,padding_right,cv2.BORDER_CONSTANT,0) 
    elif h<w:
        padding_top, padding_down = int((w-h)/2), w-h-int((w-h)/2)
        img = cv2.copyMakeBorder(img,padding_top,padding_down,0,0,cv2.BORDER_CONSTANT,0)

    assert img.shape[0] == img.shape[1]
    return cv2.resize(img, (256, 256))

def extract_f(detect_model, video_name):
    # all frame paths of the given video
    img_list = sorted(glob.glob(os.path.join("data", video_name, "*")), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    # fragment id list
    fragment = [int(i.split('/')[-1].split('.')[0]) for i in img_list]
        
    with torch.no_grad():
        V_box = detect_model.extract(img_list)
    
    crop_img_list = []
    V_box = np.array([v[0][:4] for v in V_box])
    for idx, v_box in enumerate(V_box):
        extract_v_box = v_box.tolist()
        assert len(extract_v_box) == 4 # check (4 parameters)
        
        img = cv2.imread(img_list[idx], 0)
        crop_img = crop_resize(img, extract_v_box)
        crop_img_list.append(crop_img)

    return crop_img_list, fragment