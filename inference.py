from models.vessel_region_extract import extract_frag_v_box
from models.extract_vessel import extract_f
from Deeplabv3.test import create_model
from Deeplabv3.test import frag_seg
from carotid_conformer3d.frag_cls_pre import load_model, frag_infer_mask_ori
from carotid_conformer3d.key_frag_extraction import key_frag_extract
import torch

classes_map = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}

def infer():
    # Detector
    frag_crop_img, fragment = extract_f(detect_model, video_name)
    
    # Segmentor
    frag_mask = frag_seg(seg_model, frag_crop_img, device)
    
    # Key Fragment Extractor
    key_frag_rate, key_crop, key_mask = key_frag_extract(fragment, frag_mask, frag_crop_img)

    # Classifier
    frag_pred_cls, conf = frag_infer_mask_ori(cls_model, key_mask, key_crop, key_frag_rate, device)
    
    print(f"AI predicts {classes_map[frag_pred_cls]}, with confidence = {conf}.")

if __name__=="__main__":
    device = torch.device('cuda:0')
    video_name = "video01" # e.g.

    """load models + eval mode"""
    # Detector
    print("======= loading detector =======")
    detect_model = extract_frag_v_box(model_path="", device=device)
    
    # Segmentor
    print("======= loading segmentor =======")
    seg_model = create_model(aux=False, num_classes=3, weights_path="", device=device)
   
    # Classifier
    print("======= loading classifier =======")
    cls_model = load_model(device=device)
    
    infer()
   
    