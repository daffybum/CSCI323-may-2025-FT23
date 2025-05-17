import os
import tkinter.messagebox
from contextlib import suppress
from random import choice
from time import sleep
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from typing import List, Tuple

os.environ['SDL_VIDEO_CENTERED'] = '1'

import numpy as np
import pygame

from GUI.button import Button
from GUI.camera_window import CameraWindow
from Sudoku.sudoku import Sudoku
from Image_Processing.process_image import SudokuImageProcessing

root = Tk()
root.withdraw()

BLOCK_SIZE = 40
SCREEN_WIDTH = 650
SCREEN_HEIGHT = 650
X = 0
Y = 1

class SudokuGUI:
    def __init__(self, matrix: np.ndarray, box_rows: int = 3, box_cols: int = 3):
        self.BOX_ROWS = box_rows
        self.BOX_COLS = box_cols
        self.NUM_ROWS = self.BOX_ROWS * self.BOX_COLS
        self.NUM_COLUMNS = self.BOX_ROWS * self.BOX_COLS
        self.PLAY_WIDTH = BLOCK_SIZE * self.NUM_COLUMNS
        self.PLAY_HEIGHT = BLOCK_SIZE * self.NUM_ROWS
        self.TOP_LEFT = (int((SCREEN_WIDTH - self.PLAY_WIDTH) / 2),
                         int((SCREEN_HEIGHT - self.PLAY_HEIGHT) / 2 - 80))

        self.matrix = matrix
        self.init_matrix = self.matrix.copy()
        try:
            self.solution_list = Sudoku(matrix.copy(), box_row=self.BOX_ROWS, box_col=self.BOX_COLS).get_solution()
            self.solution = self.solution_list[0]
        except Exception:
            tkinter.messagebox.showerror(title="Error",
                                         message="Solution does not exist or Image not clear, Try Again.")
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.selected_box = (0, 0)
        self.locked_pos = self.get_locked_pos()
        self.home_icon = pygame.image.load('.images/home_icon.png')
        self.button_home = Button(60, 60, 70, 70, (200, 200, 200), '  ')
        self.button_load_image = Button(162, 510, 250, 60, (200, 200, 200), "Load from File")
        self.button_solve = Button(325, 590, 250, 60, (200, 200, 200), "Solve")
        self.solver_filled_pos = []  # positions filled by solver
        self.button_back = Button(488, 510, 250, 60, (200, 200, 200), "Back to Edit")
        


    def main_menu(self):
        self.window.fill((255, 255, 255))
        font = pygame.font.SysFont('comicsans', 60)
        font2 = pygame.font.SysFont('comicsans', 30)
        label = font.render('Auto Sudoku Solver', 1, (0, 0, 0))
        label2 = font2.render('CSCI323 FT23', 1, (0, 0, 0))
        self.window.blit(label, ((SCREEN_WIDTH - label.get_width()) / 2,
                                 100 - label.get_height() / 2))
        self.button_load_image.draw(self.window)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.graceful_exit()

            if self.button_load_image.clicked(event):
                self.handle_click(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': (self.button_load_image.x, self.button_load_image.y)}))
                return 1

        pygame.display.update()
        return 0

    def play_game(self):
        self.draw_window()
        for event in pygame.event.get():
            if not self.handle_click(event):
                return False
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.graceful_exit()

            if event.type == pygame.KEYDOWN:
                box_i, box_j = self.selected_box
                if event.key == pygame.K_UP:
                    box_j -= 1
                if event.key == pygame.K_DOWN:
                    box_j += 1
                if event.key == pygame.K_RIGHT:
                    box_i += 1
                if event.key == pygame.K_LEFT:
                    box_i -= 1
                self.selected_box = (box_i % self.NUM_ROWS, box_j % self.NUM_COLUMNS)

        # get pressed keys and current selected box
        keys = pygame.key.get_pressed()
        box_i, box_j = self.selected_box

        # allow editing of any cell (no locked cells)
        for i in range(pygame.K_0, pygame.K_0 + self.NUM_ROWS + 1):
            if keys[i]:
                self.matrix[(box_j, box_i)] = i - pygame.K_0
                self.init_matrix[(box_j, box_i)] = i - pygame.K_0

        for i in range(pygame.K_KP0, pygame.K_KP0 + self.NUM_ROWS + 1):
            if keys[i]:
                self.matrix[(box_j, box_i)] = i - pygame.K_KP0
                self.init_matrix[(box_j, box_i)] = i - pygame.K_KP0

        # allow deletion for any cell
        if keys[pygame.K_DELETE]:
            self.matrix[(box_j, box_i)] = 0
            self.init_matrix[(box_j, box_i)] = 0

        return True

    def handle_click(self, event):
        if self.button_load_image.clicked(event):
            path = askopenfilename(filetypes=[("image", "*.png *.jpg *.jpeg *.bmp")])
            if len(path) != 0:
                sip = SudokuImageProcessing(fname=path)
                mat = sip.get_matrix()
                if mat is None:
                    tkinter.messagebox.showerror(title="Error", message="Unable to load file, try with different image.")
                    return
                _, (box_rows, box_cols) = sip.get_dimensions()
                self.__init__(mat, box_rows, box_cols)
                np.save('last_loaded.npy', self.matrix)
                np.save('last_loaded_dim.npy', np.array([self.BOX_ROWS, self.BOX_COLS]))
                self.play_game()

        if self.button_solve.clicked(event):
            try:
                edited_matrix = self.matrix.copy()
                sudoku_solver = Sudoku(edited_matrix, box_row=self.BOX_ROWS, box_col=self.BOX_COLS)
                solution_list = sudoku_solver.get_solution()
                if not solution_list:
                    raise ValueError("No solution found.")
                solved_matrix = solution_list[0]

                # Find which cells were filled by solver
                solver_filled = [(i, j) for i in range(self.NUM_ROWS)
                                for j in range(self.NUM_COLUMNS)
                                if edited_matrix[i, j] == 0 and solved_matrix[i, j] != 0]

                # Launch solved GUI
                solved_gui = SudokuGUI(solved_matrix, self.BOX_ROWS, self.BOX_COLS)
                solved_gui.solver_filled_pos = solver_filled
                solved_gui.locked_pos = solved_gui.get_locked_pos()

                # Show solved board with a back button
                while True:
                    solved_gui.draw_window(solved=True)
                    for event in pygame.event.get():
                        if solved_gui.button_back.clicked(event):
                            return True  # go back to editable screen
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            solved_gui.graceful_exit()

            except Exception:
                tkinter.messagebox.showerror(
                    title="Error",
                    message="This puzzle cannot be solved. Please check your input and try again."
                )
        if self.button_home.clicked(event):
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if mouse_x in range(self.TOP_LEFT[X], self.TOP_LEFT[X] + self.NUM_COLUMNS * BLOCK_SIZE) and                mouse_y in range(self.TOP_LEFT[Y], self.TOP_LEFT[Y] + self.NUM_ROWS * BLOCK_SIZE):
                self.selected_box = ((mouse_x - self.TOP_LEFT[X]) // BLOCK_SIZE,
                                     (mouse_y - self.TOP_LEFT[Y]) // BLOCK_SIZE)

        return True

    def get_locked_pos(self):
        locked_pos = []
        for i in range(self.NUM_ROWS):
            for j in range(self.NUM_COLUMNS):
                if self.matrix[i, j] != 0:
                    locked_pos.append((i, j))
        return locked_pos

    def draw_window(self, solved=False):
        self.window.fill((255, 255, 255))
        font = pygame.font.SysFont('comicsans', 48)
        label = font.render('SUDOKU', 1, (0, 0, 0))
        self.window.blit(label, (self.TOP_LEFT[X] + self.PLAY_WIDTH / 2 - (label.get_width() / 2),
                                40 - (label.get_height() / 2)))

        # Draw grid lines
        for i in range(self.NUM_ROWS + 1):
            pygame.draw.line(self.window, (0, 0, 0),
                            (self.TOP_LEFT[X], self.TOP_LEFT[Y] + i * BLOCK_SIZE),
                            (self.TOP_LEFT[X] + self.PLAY_WIDTH, self.TOP_LEFT[Y] + i * BLOCK_SIZE),
                            4 if i % self.BOX_ROWS == 0 else 1)
        for i in range(self.NUM_COLUMNS + 1):
            pygame.draw.line(self.window, (0, 0, 0),
                            (self.TOP_LEFT[X] + i * BLOCK_SIZE, self.TOP_LEFT[Y]),
                            (self.TOP_LEFT[X] + i * BLOCK_SIZE, self.TOP_LEFT[Y] + self.PLAY_HEIGHT),
                            4 if i % self.BOX_COLS == 0 else 1)

        font = pygame.font.SysFont('comicsans', 32)
        for i in range(self.NUM_ROWS):
            for j in range(self.NUM_COLUMNS):
                if self.matrix[i, j] == 0:
                    continue
                # Color logic
                if solved and hasattr(self, "solver_filled_pos") and (i, j) in self.solver_filled_pos:
                    num_color = (128, 193, 42)  # Solver-filled: green
                else:
                    num_color = (0, 0, 0)  # User-edited: blue

                label = font.render(str(self.matrix[i, j]), 1, num_color)
                self.window.blit(label,
                                (self.TOP_LEFT[X] + j * BLOCK_SIZE - label.get_width() / 2 + BLOCK_SIZE / 2,
                                self.TOP_LEFT[Y] + i * BLOCK_SIZE - label.get_height() / 2 + BLOCK_SIZE / 2))

        # Draw selector box
        pygame.draw.rect(self.window, (100, 178, 255),
                        (self.TOP_LEFT[X] + self.selected_box[0] * BLOCK_SIZE,
                        self.TOP_LEFT[Y] + self.selected_box[1] * BLOCK_SIZE,
                        BLOCK_SIZE, BLOCK_SIZE), 4)

        # Draw buttons
        self.button_home.draw(self.window)
        self.window.blit(self.home_icon,
                        (self.button_home.x - self.home_icon.get_width() / 2,
                        self.button_home.y - self.home_icon.get_height() / 2))
        self.button_load_image.draw(self.window)
        self.button_solve.draw(self.window)

        # Only show "Back" button on solved screen
        if solved and hasattr(self, "button_back"):
            self.button_back.draw(self.window)

        pygame.display.update()


    def graceful_exit(self):
        np.save('last_loaded.npy', self.init_matrix)
        np.save('last_loaded_dim.npy', np.array([self.BOX_ROWS, self.BOX_COLS]))
        pygame.quit()
        quit()
