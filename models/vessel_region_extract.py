import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

imgsz = 416
conf_thres = 0.01  # 0.001
iou_thres = 0.45  # 0.6 for NMS

class ModelWrapper(torch.nn.Module):
    def __init__(self, models):
        super(ModelWrapper, self).__init__()
        self.model1 = models[0]
    def forward(self, x):
        out1 = self.model1(x)
        return (out1,)

class extract_frag_v_box():
    def __init__(self, model_path, device):
        model, half = self.create_model(model_path, device)
        self.device = device
        self.model = model
        self.half = half

    def extract(self, img_list):
        new_pre_C = self.video_infer(img_list, self.model, self.half, self.device)
        return new_pre_C

    def create_model(self, model_path, device):
        # Load model
        m = attempt_load(model_path, map_location=device)  # load FP32 model

        model = ModelWrapper([m])
        # Half
        half = device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()
        model.eval()
        return model, half

    def video_infer(self, img_list, model, half, device):
        new_pre_C = []
        for image_path in img_list:
            # read img
            img_grey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            imgRGB = np.zeros(shape=(img_grey.shape[0], img_grey.shape[1], 3), dtype=np.uint8)
            imgRGB[:, :, 0] = img_grey
            imgRGB[:, :, 1] = img_grey
            imgRGB[:, :, 2] = img_grey
            img_ori = imgRGB

            ori_shape = img_ori.shape[:2]  # original hw
            infer_shape = [imgsz, imgsz]

            # pre-processing
            img = cv2.resize(img_ori, (imgsz, imgsz))
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img[np.newaxis, :, :, :]) 

            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            # Run model
            outs = model(img)  # inference and training outputs
            [inf_out1, train_out1] = outs[0]

            # Run NMS
            # output: n，6 (x1, y1, x2, y2, conf, cls)
            output1 = non_max_suppression(inf_out1, conf_thres=conf_thres, iou_thres=iou_thres, labels=[])

            # Save
            out1 = output1[0].cpu()

            # post-processing
            # vessel extraction
            pre_c1 = out1[out1[:, 5] == 0]
            pre_c1 = pre_c1[0:2]
            pre_c1 = pre_c1[pre_c1[:, 4] > 0.2]
            if pre_c1.size(0) > 0:
                pre_c1[:, 0:4] = scale_coords(infer_shape, pre_c1[:, 0:4], ori_shape)
            pre_c1 = pre_c1.numpy().tolist()
            new_pre_C.append(pre_c1)

        assert len(new_pre_C) == len(img_list)
        return new_pre_C