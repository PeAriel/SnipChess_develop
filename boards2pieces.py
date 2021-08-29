import os
import cv2
import numpy as np
from random import randint
from fen2png import PIECES_DICT, is_int
from tqdm import tqdm


def extand_name(fen_list):
    """
    extends a fen name to be 8 characters long for each row, for easy counting.
    :param fen_list: list of strings where each string represents a rank position
    """
    for i in range(8):
        jdx = 0
        for _ in range(len(fen_list[i])):
            if is_int(fen_list[i][jdx]):
                nempty = int(fen_list[i][jdx])
                fen_list[i] = fen_list[i][:jdx] + 'e' * nempty + fen_list[i][jdx + 1:]
                jdx += nempty
                continue
            jdx += 1
    return fen_list

def board2peices(path, reduce=None, sft=20):
    PIECES_DICT['e'] = 'e'
    """
    given the *absolute* path to the folder containing the boards, this function writes
    each square as a separate png. It also adds a random shift to the squares to include
    some data from adjacent squares. To disable the shift, input sft=0.
    :parm reduce: int. Limits the number of black and white pawns, as well as empty
                  squares (from all boards) to this number.
    """
    boards_list = os.listdir(path)
    if not os.path.isdir(path + '/pieces'):
        os.mkdir(path + '/pieces')

    for b, board in enumerate(tqdm(boards_list)):
        count_dict = {'e': 0, 'bp': 0, 'wp': 0}
        if board[0] == '.' or board == 'pieces':
            continue
        name = extand_name(board.split('.')[0].split('S'))
        img = cv2.imread(path + '/' + board)
        board_length = len(img[:, :, 0])
        square_size = board_length // 8
        for i in range(8):
            for j in range(8):
                shift = randint(0, sft)  # random global shift for the square
                randx = randint(0, 1)  # random indicator when x is shifted
                randy = randint(0, 1)  # random indicator when y is shifted
                xi = square_size * j + shift * randx
                xf = square_size * (j + 1) + shift * randx
                yi = square_size * i + shift * randy
                yf = square_size * (i + 1) + shift * randy
                x = np.array([s for s in range(xi, xf)]) % board_length  # periodic boundary conditions for the shift
                y = np.array([s for s in range(yi, yf)]) % board_length  # periodic boundary conditions for the shift
                piece_img = img[np.ix_(y, x)]
                piece_img = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)  # convert to gray scale

                piece_name = PIECES_DICT.get(name[i][j])
                if reduce:
                    if piece_name in count_dict:
                        count_dict[piece_name] += 1
                        if count_dict[piece_name] <= reduce:
                            cv2.imwrite(path + '/pieces/%s_%d%d%d.png' % (piece_name, b, i, j), piece_img)
                        else:
                            continue
                cv2.imwrite(path + '/pieces/%s_%d%d%d.png' % (piece_name, b, i, j), piece_img)

def main():
    training_path = os.getcwd() + '/resources/training_dataset'
    validation_path = os.getcwd() + '/resources/validation_dataset'
    board2peices(training_path, reduce=3)
    board2peices(validation_path, reduce=4)



if __name__ == '__main__':
    main()