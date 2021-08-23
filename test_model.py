#!/Library/Frameworks/Python.framework/Versions/3.9/bin/python3

from model import ChessConvNet
import os, sys
import cv2
import torch
import numpy as np
from time import time

labels_dict = {'e': 0,
               'br': 1,
               'bb': 2,
               'bn': 3,
               'bq': 4,
               'bk': 5,
               'bp': 6,
               'wr': 7,
               'wb': 8,
               'wn': 9,
               'wq': 10,
               'wk': 11,
               'wp': 12}
labels_list = [k for k,v in labels_dict.items()]

def ss_regions(cvimage, method='f', verbosity=True):

    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(cvimage)

    if 'f' in method:
        ss.switchToSelectiveSearchFast()
        if verbosity:
            print("[INFO] using *fast* selective search")
    elif 'q' in method:
        ss.switchToSelectiveSearchQuality()
        if verbosity:
            print("[INFO] using *quality* selective search")

    # run selective search on the input image
    start = time()
    if verbosity:
        print("\n===== Starting selective search =====")
    rects = ss.process()
    if verbosity:
        print("===== Finished selective search =====\n")
    end = time()

    if verbosity:
        print("[INFO] selective search took {:.4f} seconds".format(end - start))
        print("[INFO] {} total region proposals".format(len(rects)))

    '''
    Here I pick only the regions which deviates from being squares at most by 2. Also, The regions'
    dimensions are bounded by the length of the image divided by 8, so I filter regions larger
    than that either.
    '''
    new_rects = []
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        if (w - h) < 3 and w <= cvimage.shape[0] // 8 and h <= cvimage.shape[0] // 8:
            new_rects.append([x, y, w, h])
    new_rects = np.array(new_rects)

    '''
    If the image of the board is ideal, we should get 64 regions if we take the maximal size regions 
    (under the above conditions). 
    If there are less than 64, than the image is not ideal and the squares will be built from the topmost
    corner manually (hoping it will work).

    After a second thought, only the manual scheme is being used. There are a few commented lines
    including a big if else that utilizes the ideal scheme. Uncomment if needed in the future.
    '''
    # max_w = np.max(new_rects[:, 2])
    # fixed_rects = new_rects[new_rects[:, 2] >= max_w, :]
    # if len(fixed_rects) < 64:
    # del fixed_rects
    fixed_rects = np.array([[0 for _ in range(4)] for _ in range(64)])

    min_indices = np.argmin(new_rects, axis=0)
    xp, _, wp, hp = new_rects[min_indices[0]]  # Here I (arbitrarily) use x as the min. Could use y instead.
    box_size = max(wp, hp)

    idx = 0
    while True:
        '''
        here I check if the distance of the last square from the edge of the board 
        (which is the edge of the image minus xp) is smaller than a treshold,
        otherwise adjust the box size
        '''
        condition = xp + box_size * 8 - (cvimage.shape[0] - xp)
        if abs(condition) > 3:
            box_size -= 1 * np.sign(condition)
            idx += 1
            if idx % 10 == 0:
                xp += 2
            if verbosity:
                print('Margin deviation = %d. Now box size = %d' % (condition, box_size))
                if idx % 10 == 0:
                    print('Reached %d iterations. Now xp = %d' % (idx, xp))
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

def visualize_regions(cvimage, regions):
    while True:
        # create a copy of original image
        imOut = cvimage.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(regions):
            # draw rectangle for region proposal till numShowRects
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

        # show output
        cv2.imshow("Output", imOut)

        # record key press
        k = cv2.waitKey(0) & 0xFF
        # q is pressed
        if k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()

def regions2squares(cvimage, regions, square_size=80):
    cvimage = cv2.cvtColor(cvimage, cv2.COLOR_BGR2GRAY)

    squares = np.zeros([64, square_size, square_size], dtype=np.uint8)
    for i in range(64):
        imOut = cvimage.copy()
        x, y, w, h = regions[i]
        squares[i, :, :] = cv2.resize(imOut[y:y+w, x:x+w], (square_size, square_size))

    return squares

MODEL_PATH = os.getcwd() + '/trained_model.pt'
model = ChessConvNet()
model.load_state_dict(torch.load(MODEL_PATH), strict=False)
model.eval()

img_path = sys.argv[1]
img = cv2.imread(img_path)

new_size = 640
img = cv2.resize(img, (new_size, new_size))
regions = ss_regions(img)
visualize_regions(img, regions)
squares = regions2squares(img, regions, square_size=80)


pred_board = [['' for _ in range(8)] for _ in range(8)]
idx = 0
for i in range(8):
    for j in range(8):
        square = np.stack((squares[idx], ) * 3, axis=0)
        prediction = model(torch.FloatTensor(square).unsqueeze(0))
        pred_board[i][j] = labels_list[torch.argmax(prediction, dim=1)]
        idx += 1

print('\n\nPredicted board:')
print('------------------------')
for file in pred_board:
    print('\t'.join(file))
