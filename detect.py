import cv2
import numpy as np
from utils.augmentations import letterbox
import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression)
from utils.torch_utils import select_device

classes = [2, 1, 50, 20, 10, 5]


@torch.no_grad()
def run(image,
        weights=ROOT / 'coin_detector.pt',  # model.pt path(s)
        data=ROOT / 'data/coin.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.45,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=25,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        ):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights=weights, device=device, data=data)
    imgsz = check_img_size(imgsz, s=model.stride)  # check image size
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    img = letterbox(image, stride=model.stride)[0]
    # Convert
    im = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]
    pred = model(im)
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
    preds = []
    for det in pred:
        try:
            box = det[0].tolist()
            box[-1] = classes[int(box[-1])]
            preds.append(box)
        except Exception:
            pass
    return preds

