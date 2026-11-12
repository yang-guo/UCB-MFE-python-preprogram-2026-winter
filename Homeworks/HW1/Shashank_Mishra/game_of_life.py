def count_live_neighbors(board, row, col):
    """Count the live cells among the (up to) 8 neighbors of board[row][col].

    Cells outside the board are treated as dead, so edges and corners
    simply have fewer neighbors to check.
    """
    n_rows = len(board)
    n_cols = len(board[0])
    count = 0
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue  # skip the cell itself
            r, c = row + d_row, col + d_col
            if 0 <= r < n_rows and 0 <= c < n_cols:
                count += board[r][c]
    return count


def evolve(initial_state):
    """Return the next generation of a Game of Life board.

    The board is a list of lists of 0 (dead) and 1 (alive). The input is not
    modified; a new board is built from the old state, so every cell is
    updated simultaneously.

    Rules (https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life):
    - A live cell with 2 or 3 live neighbors survives.
    - A dead cell with exactly 3 live neighbors becomes alive.
    - Every other cell is dead in the next generation.
    """
    if not initial_state or not initial_state[0]:
        return [row[:] for row in initial_state]

    next_state = []
    for row in range(len(initial_state)):
        new_row = []
        for col in range(len(initial_state[0])):
            neighbors = count_live_neighbors(initial_state, row, col)
            alive = initial_state[row][col] == 1
            if alive and neighbors in (2, 3):
                new_row.append(1)
            elif not alive and neighbors == 3:
                new_row.append(1)
            else:
                new_row.append(0)
        next_state.append(new_row)
    return next_state


test_case_1 = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
]

test_case_2 = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
]

test_case_2_next = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]


assert evolve(test_case_1) == test_case_1
assert evolve(test_case_2) == test_case_2_next

print("All test cases passed!")
