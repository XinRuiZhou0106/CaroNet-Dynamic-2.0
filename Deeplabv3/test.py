import sys
import os
import cv2
import numpy as np
import torch
sys.path.append(os.path.join(os.getcwd(), "Deeplabv3"))
from model import deeplabv3_mobilenetv3_large
from skimage import morphology

def create_model(aux, num_classes, weights_path, device):
    model = deeplabv3_mobilenetv3_large(aux=aux, num_classes=num_classes)
    if weights_path:
        weights_dict = torch.load(weights_path, map_location='cpu')['model']
        model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()
    return model

def normalize(im):
    """
    Normalize volume's intensity to range [0, 1], for suing image processing
    Compute global maximum and minimum cause cerebrum is relatively homogeneous
    """
    mean = np.mean(im)
    std = np.std(im)
    if std == 0:
        std = 1
    return (im - mean) / std

def mask_postprocess(img):
    # retaining the largest connected component (residual lumen only)
    if np.count_nonzero(img == 127) != 0:
        vessel_img = img.copy()
        vessel_img[vessel_img == 127] = 255
        img[img == 255] = 0
        img[img == 127] = 255
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        max_idx = np.argmax(area)
        for j in range(len(contours)):
            if j != max_idx:
                cv2.fillPoly(img, [contours[j]], 0)
        img = img.astype(np.int64) + vessel_img.astype(np.int64)
        img[img == 510] = 127
        img = img.astype(np.uint8)
    return img

def mask_postprocess_fill_hole(img):
    # remove holes
    # processing residual lumen
    vessel_img = img.copy()
    vessel_img[vessel_img == 127] = 255
    img[img == 255] = 0
    img[img == 127] = 255 
    pre_mask_rever = img<=0
    pre_mask_rever = morphology.remove_small_objects(pre_mask_rever, min_size=700)
    img[pre_mask_rever<=0] = 255
    img = img.astype(np.int64) + vessel_img.astype(np.int64)
    img[img > 255] = 127
    img = img.astype(np.uint8)

    # processing vessel
    vessel_img = img.copy()
    vessel_img[vessel_img == 127] = 255
    img[img == 255] = 0
    img[img == 127] = 255 
    pre_mask_rever = vessel_img<=0 
    pre_mask_rever = morphology.remove_small_objects(pre_mask_rever, min_size=700)
    vessel_img[pre_mask_rever<=0] = 255
    vessel_img = img.astype(np.int64) + vessel_img.astype(np.int64)
    vessel_img[vessel_img > 255] = 127
    vessel_img = vessel_img.astype(np.uint8)
    
    return vessel_img

def frag_seg(model, frag_crop_img, device):
    one_frag_masks = evaluate(model, frag_crop_img, device)
    return one_frag_masks

def evaluate(model, frag_crop, device):
    labels = [0,1,2]
    palettes=[0,127,255] # 3 classes
    one_frag_masks = []
    for img_crop in frag_crop:
        # pre-processing
        img_crop = np.array(img_crop,dtype=np.float32)
        img_crop = img_crop/255.0
        img_crop = normalize(img_crop)
        img_crop = torch.from_numpy(img_crop)
        # expand batch dimension
        img_crop = torch.unsqueeze(torch.unsqueeze(img_crop, dim=0), dim=0)
        image = img_crop.to(device)
        # infer
        with torch.no_grad():
            output = model(image.to(device))
            prediction = torch.softmax(output['out'].squeeze(0),dim=0)
            prediction = prediction.argmax(0)
            prediction = prediction.cpu().numpy().astype(np.uint8)
            for label,palette in zip(labels,palettes):
                prediction[prediction==label] = palette
            # mask post-processing
            one_frag_masks.append(mask_postprocess_fill_hole(mask_postprocess(prediction)))
    return one_frag_masks
            
        
