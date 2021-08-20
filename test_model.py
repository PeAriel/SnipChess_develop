#!/Library/Frameworks/Python.framework/Versions/3.9/bin/python3

from model import ChessConvNet
import os, sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

def ss_regions(image):
    cvimage = cv2.imread(image)
    new_height = 640
    new_width = int(cvimage.shape[1] * new_height / cvimage.shape[0])
    cvimage = cv2.resize(cvimage, (new_width, new_height))

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(cvimage)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    new_rects = []
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        if (w - h) < 3 and w <= cvimage.shape[0] // 8 and h <= cvimage.shape[0] // 8:
            new_rects.append([x, y, w, h])
    new_rects = np.array(new_rects)

    fixed_rects = np.array([[0 for _ in range(4)] for _ in range(64)])

    min_indices = np.argmin(new_rects, axis=0)
    xp, _, wp, hp = new_rects[min_indices[0]]  # Here I (arbitrarily) use x as the min. Could use y instead.
    box_size = max(wp, hp)

    idx = 0
    while True:
        condition = xp + box_size * 8 - (cvimage.shape[0] - xp)
        if abs(condition) > 3:
            box_size -= 1 * np.sign(condition)
            idx += 1
            if idx % 10 == 0:
                xp += 2
            continue
        else:
            break
        # Now construct the new 64 fixed rects
    idx = 0
    for i in range(8):
        for j in range(8):
            fixed_rects[idx] = [xp + box_size * j, xp + box_size * i, box_size, box_size]
            idx += 1

    return fixed_rects

def regions2squares(image, regions, square_size=80):
    squares = np.zeros([64, square_size, square_size])
    for i, region in enumerate(regions):
        x, y, w, h = region
        square = image[x]

MODEL_PATH = os.getcwd() + '/trained_model_classifier.pt'
model = ChessConvNet()
model.load_state_dict(torch.load(MODEL_PATH), strict=False)
model.eval()

img_path = sys.argv[1]
img = cv2.imread(img_path)
tensor_image = torch.FloatTensor(img)

