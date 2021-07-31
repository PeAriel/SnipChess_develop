from fen2png import is_int

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


def fens2representations(fens):
    """
    :param fens: a single fen(string) or a list of fens. Note: input string must not have the additional information
                 on the position
    :returns:    an ascii representation of the board. It is a list 8*8+7 entries, where each number is the
                 ascii representation of the character of the fen, 0 means an empty square, and the separators are
                 identified by the ascii representation of the letter S.
    """
    is_list = (len(fens[0]) > 1)
    if not is_list:
        fens = [fens]
    ascii0 = ord('0')
    ascii_rep_list = [[] for _ in range(len(fens))]
    for j, fen in enumerate(fens):
        ascii_rep = [0 for _ in range(8 ** 2 + 7)]  # 8 by 8 grid plus 7 separators
        pos = 0
        for char in fen:
            if is_int(char):
                pos += (ord(char) - ascii0)
                continue
            ascii_rep[pos] = ord(char)
            pos += 1
        ascii_rep_list[j] = ascii_rep

    if not is_list:
        return ascii_rep_list[0]
    return ascii_rep_list

def remove_suffix(file_list):
    """
    :param file_list: file list in a folder
    :returns: the same list, where the suffix denoting the file type is removed
    """
    for i, name in enumerate(file_list):
        file_list[i] = name.split('.')[0]

    return file_list

class ChessBoardsDataset(Dataset):
    """
    Chess boards data set object.
    Uses ascii representation to represent the fen as an array of integers:
    ord(char) ------> ascii integer
    chr(int)  ------> char string

    each label has the same length.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.fens_png = os.listdir(data_dir)
        fen_names = remove_suffix(self.fens_png)
        self.targets = fens2representations(fen_names)

    def __len__(self):
        return len(self.fens_png)

    def __getitem__(self, idx):
        img_path = self.data_dir + '/' + self.fens_png[idx] + '.png'
        img = Image.open(img_path)

        x = transforms.ToTensor()(img)
        y = torch.tensor(self.targets[idx], dtype=torch.float)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        return x, y