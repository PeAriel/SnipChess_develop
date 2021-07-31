from fen2png import DrawBoard, RESOURCES
import os
from random import randint

FENS = os.getcwd() + '/' + RESOURCES + 'fen_files/'

def random_ints_list(max_int, length):
    rand_ints = [0 for _ in range(length)]
    for i in range(length):
        rand_ints[i] = randint(0, max_int - 1)

    return rand_ints

def generate_pngs(number_of_pngs, themes, train=True):
    if train:
        train_test = 'training'
    else:
        train_test = 'validation'

    if type(themes) is str:
        themes = [themes]

    with open(FENS + 'unique00.fen', 'r') as f:
        print('Starting to read fen file.')
        file_lines = f.read().splitlines()
        print('Finished reading fen file.')
    num_lines = len(file_lines)
    random_lines = random_ints_list(num_lines, number_of_pngs)

    for theme in themes:
        for line in random_lines:
            fen = file_lines[line].split()[0]
            board = DrawBoard(fen, theme)
            output_path = os.getcwd() + '/resources/{}_dataset/{}'.format(train_test, 'S'.join(fen.split('/')))
            board.save_board(output_path)

def gen_starting_position(output_path):
    board = DrawBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', 'classic', square_size=40)
    board.save_board(output_path)


def main():
    numbers_of_training_pngs = int(1000)
    numbers_of_validation_pngs = int(200)
    theme = ['classic']
    generate_pngs(numbers_of_training_pngs, theme)
    generate_pngs(numbers_of_validation_pngs, theme, train=False)

if __name__ == '__main__':
    main()