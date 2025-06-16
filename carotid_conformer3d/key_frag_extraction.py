import numpy as np
import torch
import cv2
import numpy as np

# get area stenosis rates of a fragment
def compute_stenosis_rate(img):
    re_vessel_img = img.copy()
    vessel_img = img.copy()
    vessel_img[vessel_img == 127] = 255
    re_vessel_img[re_vessel_img == 255] = 0
    re_vessel_img[re_vessel_img == 127] = 255
    vessel_contour, _ = cv2.findContours(vessel_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    re_vessel_contour, _ = cv2.findContours(re_vessel_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    v_areas = []
    re_v_areas = []
    for v in vessel_contour:
        v_areas.append(cv2.contourArea(v))
    for re_v in re_vessel_contour:
        re_v_areas.append(cv2.contourArea(re_v))

    if len(re_v_areas) == 0:
        area_re_vessel = 0.0
    else:
        area_re_vessel = max(re_v_areas)
    
    area_vessel = max(v_areas)
    stenosis_rate = 1 - (area_re_vessel/area_vessel)
    return stenosis_rate

# get area stenosis rates of each fragment
def get_stenosis_rate(f_mask):
    frag_stenosis_rate = []
    for mask in f_mask:
        stenosis_rate = compute_stenosis_rate(mask)
        frag_stenosis_rate.append(stenosis_rate)
    max_frame = [frag_stenosis_rate.index(max(frag_stenosis_rate)), max(frag_stenosis_rate)]  # the max rate frame id

    return max_frame, frag_stenosis_rate

def get_key_frames_test(fragment, max_ai_id, sample_interval_list: list):
    min_len = 8 
    
    key_fragment = {}
    for x in sample_interval_list:
        key_fragment.update({x: []})
        
    for sample_interval in sample_interval_list:
        sample_rate = sample_interval + 1
        
        # sampling at a fixed interval
        left_frame_id = [fragment[max_ai_id - (i + 1) * sample_rate] for i in range(max_ai_id // sample_rate)]
        left_frame_id.sort()
        right_frame_id = [fragment[max_ai_id + (i + 1) * sample_rate] for i in range((len(fragment) - max_ai_id - 1) // sample_rate)]
        key_id = left_frame_id + [fragment[max_ai_id]] + right_frame_id
        
        # continue condensing if the length after sampling still exceeds the minimum threshold
        if len(key_id) >= min_len:
            max_id_new = key_id.index(fragment[max_ai_id])
            th = (min_len - 1)/2
            left_frame_num, right_frame_num = len(left_frame_id), len(right_frame_id)
            if left_frame_num <= th: 
                key = key_id[:min_len]
            if right_frame_num <= th:
                need_left_num = min_len - 1 - right_frame_num
                key = key_id[max_id_new - need_left_num :]
            if left_frame_num > th and right_frame_num > th: 
                need_left_num = int(th) 
                need_right_num = min_len - need_left_num - 1
                key = key_id[max_id_new - need_left_num : max_id_new + need_right_num + 1]
            assert len(key) == min_len
            key_fragment[sample_interval] = key
    
    print("AI Max Frame: ", fragment[max_ai_id])
    print("Ori Frag: ", fragment)
    print("Key Frag: ", key_fragment)
    
    return key_fragment

def key_frag_extract(fragment, frag_mask, frag_crop_img):
    max_frame, frag_stenosis_rate = get_stenosis_rate(frag_mask) 
    max_ai_id = max_frame[0]
    key_fragment_dict = get_key_frames_test(fragment, max_ai_id, [2])

    key_fragment = key_fragment_dict[2]
    key_frag_rate = [frag_stenosis_rate[i - fragment[0]] for i in key_fragment] # area reduction of each frame in the key fragment
    key_frag_rate = torch.from_numpy(np.array(key_frag_rate, dtype=np.float32))
    key_mask = [frag_mask[j - fragment[0]] for j in key_fragment]
    key_crop = [frag_crop_img[k - fragment[0]] for k in key_fragment]

    assert len(key_mask) == len(key_frag_rate) == len(key_crop)
    return key_frag_rate, key_crop, key_mask