import os
from random import randint


def random_ints_list(max_int, length):
    rand_ints = [0 for _ in range(length)]
    for i in range(length):
        rand_ints[i] = randint(0, max_int - 1)

    return rand_ints

def main():
    file_name = 'unique00.fen'
    with open(file_name, 'r') as f:
        print('Started reading large fen file %s...' % file_name)
        fen_data = f.read().splitlines()
        print('Finished reading.')

    file_length = len(fen_data)
    num_of_fens_to_generate = int(1e6)
    random_ints = random_ints_list(file_length, num_of_fens_to_generate)

    with open('smaller_fen.txt', 'w') as f:
        for line in random_ints:
            f.write(fen_data[line])
            f.write('\n')


if __name__ == '__main__':
    main()