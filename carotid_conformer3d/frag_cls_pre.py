import os
import sys
import torch
import math
sys.path.append(os.path.join(os.getcwd(), "carotid_conformer3d"))
from conformer3d import Conformer3d
import numpy as np

def load_model(device):
    model = Conformer3d(model_arch='carotid_conformer3d/model.yaml', patch_size=16, in_chans=2, num_classes=2, base_channel=64, 
                        channel_ratio=4, num_med_block=0, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                        qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
    model.to(device)
    model.eval()
    return model

def entropy(probabilities):
    entropy_value = 0.0
    for p in probabilities:
        if p > 0:
            entropy_value += -p * math.log2(p)
    return entropy_value

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def frag_infer_mask_ori(model, key_frag_mask, key_frag_ori, key_frag_rate, device):

    def com_op(frag_mask, frag_ori, frag_ste_rate):
        I = np.ones((256, 256), dtype=np.float32)
        # common operation
        frag_mask = [frag_m.astype(np.float32)/255.0 for frag_m in frag_mask]
        # attention target
        frag_ori = [(I + frag_m) * frag_o for (frag_m, frag_o) in zip(frag_mask, frag_ori)]
        frag_ori = [normalize(frag_o) for frag_o in frag_ori] 
        # introducing prior info (area reduction)
        frag_ori = prior_rate_info(frag_ori, frag_ste_rate) 
        frag_ori = torch.stack(frag_ori,dim=0)
        frag_ori = frag_ori.permute(3,0,1,2) # c(2), frag_len, 256, 256
        frag_ori = frag_ori.unsqueeze(0)
        assert frag_ori.shape == (1, 2, 8, 256, 256)
        return frag_ori
    
    # data 
    key_frag_mask, key_frag_ori = np.stack(key_frag_mask,axis=0), np.stack(key_frag_ori,axis=0)
    key_frag = com_op(key_frag_mask, key_frag_ori, key_frag_rate)
    key_frag = key_frag.to(device)
    
    # infer
    with torch.no_grad():
        out = model(key_frag)
    predicted_labels = torch.softmax(out,dim=-1).detach().argmax(dim=1)

    prob = torch.softmax(out,dim=-1).detach().cpu().numpy()
    um = entropy(prob[0]) # AI prediction uncertainty

    return int(predicted_labels.cpu().numpy().squeeze(0)), sigmoid(1-um)

def normalize(im):
    """
    Normalize volume's intensity to range [0, 1], for suing image processing
    Compute global maximum and minimum cause cerebrum is relatively homogeneous
    """
    mean = np.mean(im)
    std = np.std(im)
    if std == 0:
        std = 1
    gray_im = (im - mean) / std
    return torch.from_numpy(gray_im[:, :, np.newaxis])

def prior_rate_info(frag, rate):
    """
    Introduce the prior stenosis area rate to guide the network to learn the spatial rate information
    """
    new_frag = []
    for im, rate_i in zip(frag,rate):
        rate_i = rate_i.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        rate_i = rate_i.repeat(im.size(0),im.size(1),1)
        im = torch.cat((im,rate_i),dim=-1)
        new_frag.append(im)
    return new_frag

