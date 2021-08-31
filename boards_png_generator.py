from fen2png import DrawBoard, RESOURCES
import os
from random import randint
from tqdm import tqdm

FENS = os.getcwd() + '/' + RESOURCES + 'fen_files/'

def random_ints_list(max_int, length):
    rand_ints = [0 for _ in range(length)]
    for i in range(length):
        rand_ints[i] = randint(0, max_int - 1)

    return rand_ints

def generate_pngs(themes, train=None, valid=None, square_size=80, verbosity=False):
    if type(themes) is str:
        themes = [themes]

    with open(FENS + 'smaller_fen.fen', 'r') as f:
        print('Starting to read fen file.')
        file_lines = f.read().splitlines()
        print('Finished reading fen file.')
    num_lines = len(file_lines)

    for theme in themes:
        if train:
            random_lines_train = random_ints_list(num_lines, train // len(themes))
            if verbosity:
                random_lines_train = tqdm(random_lines_train, desc='Training images completed of theme {}'.format(theme))
            for line in random_lines_train:
                fen = file_lines[line].split()[0]
                board = DrawBoard(fen, theme, square_size)
                output_path = os.getcwd() + '/resources/training_dataset/{}'.format('S'.join(fen.split('/')))
                board.save_board(output_path)
        if valid:
            random_lines_valid = random_ints_list(num_lines, valid // len(themes))
            if verbosity:
                random_lines_valid = tqdm(random_lines_valid, desc='Validation images completed of theme {}'.format(theme))
            for line in random_lines_valid:
                fen = file_lines[line].split()[0]
                board = DrawBoard(fen, theme, square_size)
                output_path = os.getcwd() + '/resources/validation_dataset/{}'.format('S'.join(fen.split('/')))
                board.save_board(output_path)

def gen_starting_position(output_path, square_size=80):
    board = DrawBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', 'classic', square_size)
    board.save_board(output_path)


def main():
    # glass is confusing!
    # theme = ['bases', 'blues', 'book', 'book2',
    #          'brown', 'classic', 'bubble_gum', 'cosmos',
    #          'dash', 'game_room', 'light', 'lolz',
    #          'tournament', 'opra', 'lichess',
    #          'icy_sea', 'sky', 'walnut', 'gothic',
    #          'standard', 'nature', '8bit']
    theme = ['icy_sea']
    generate_pngs(theme, train=10, valid=2, verbosity=True)
    # small square size will be inflated so it pixelized. Need to generate some of those
    generate_pngs(theme, train=10, valid=2, verbosity=True, square_size=20)

if __name__ == '__main__':
    main()