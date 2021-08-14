import cv2
import argparse
from time import time
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Selective search using OpenCv.')
    parser.add_argument('-i', '--image',
                        dest='image',
                        type=str,
                        required=True,
                        help='Path to the image.')
    parser.add_argument('-m', '--method',
                        dest='method',
                        required=False,
                        type=str,
                        choices=['fast', 'f', 'quality', 'q'],
                        help='Method to be used, either fast or quality.')
    parser.add_argument('-v', '--verbosity',
                        dest='verbosity',
                        type=int,
                        choices=[0, 1],
                        required=False,
                        help='Increase verbosity level.')
    parser.add_argument('--visualize',
                        dest='visualize',
                        type=int,
                        choices=[0, 1],
                        required=False,
                        help='Visualize generated regions.')

    args = parser.parse_args()

    method = 'f'
    if args.method:
        method = args.method

    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    cvimage = cv2.imread(args.image)
    new_height = 400
    new_width = int(cvimage.shape[1]*new_height/cvimage.shape[0])
    cvimage = cv2.resize(cvimage, (new_width, new_height))

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(cvimage)

    if 'f' in method:
        ss.switchToSelectiveSearchFast()
        if args.verbosity:
            print("[INFO] using *fast* selective search")
    elif 'q' in method:
        ss.switchToSelectiveSearchQuality()
        if args.verbosity:
            print("[INFO] using *quality* selective search")

    # run selective search on the input image
    start = time()
    if args.verbosity:
        print("\n===== Starting selective search =====")
    rects = ss.process()
    if args.verbosity:
        print("===== Finished selective search =====\n")
    end = time()

    if args.verbosity:
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
            box_size -=  1*np.sign(condition)
            idx += 1
            if idx % 10 == 0:
                xp += 2
            if args.verbosity:
                print('Margin deviation = %d. Now box size = %d' % (condition, box_size))
                if idx % 10 == 0:
                    print('Reached 10 iterations. Now xp = %d' % xp)
            continue
        else:
            break
        # Now construct the new 64 fixed rects
    idx = 0
    for i in range(8):
        for j in range(8):
            fixed_rects[idx] = [xp + box_size * j, xp + box_size * i, box_size, box_size]
            idx += 1
    # else:
    #     # TODO arrange the 64 bounding boxes in the desired order for the classification stage
    #     pass




    if args.visualize:
        # number of region proposals to show
        numShowRects = 64
        # increment to increase/decrease total number
        # of reason proposals to be shown
        increment = 50
        while True:
            # create a copy of original image
            imOut = cvimage.copy()

            # itereate over all the region proposals
            for i, rect in enumerate(fixed_rects):
                # draw rectangle for region proposal till numShowRects
                if (i < numShowRects):
                    x, y, w, h = rect
                    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    break

            # show output
            cv2.imshow("Output", imOut)

            # record key press
            k = cv2.waitKey(0) & 0xFF

            # m is pressed
            if k == 109:
                # increase total number of rectangles to show by increment
                numShowRects += increment
            # l is pressed
            elif k == 108 and numShowRects > increment:
                # decrease total number of rectangles to show by increment
                numShowRects -= increment
            # q is pressed
            elif k == 113:
                break
        # close image show window
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()