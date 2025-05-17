
from os.path import exists
import numpy as np
import pygame
from game_window import SudokuGUI

# initialize the pygame runtime
pygame.init()
# set caption for game window
pygame.display.set_caption('CSCI323 Sudoku Solver')


def main_game():
    """main method to be called"""

    # infinte loop
    while True:
        # if last loaded puzzle exists
        if exists('last_loaded.npy') and exists('last_loaded_dim.npy'):
            # load the last loaded puzzle
            mat = np.load('last_loaded.npy')
            # load the last loaded dimensions
            box_rows, box_cols = tuple(np.load('last_loaded_dim.npy'))
        else:
            # load the sample puzzle
            mat = np.load('sample.npy')
            # load the sample puzzle dimensions
            box_rows, box_cols = (3, 3)
        # creat an instance of the GUI for the game
        sg = SudokuGUI(mat.copy(), box_rows, box_cols)

        # initial value for the menu option
        menu_val = 0
        # run while the main_menu doesnt generate a valid option
        while not menu_val:
            menu_val = sg.main_menu()

        # if play game option is selected
        if menu_val == 1:
            # play game while home button is not clicked
            while sg.play_game():
                pass
        # if augmented reality option is selected


if __name__ == '__main__':
    main_game()
