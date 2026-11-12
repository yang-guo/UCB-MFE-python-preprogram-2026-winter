def count_live_neighbors(board, row, col):
    """Count the live cells among the (up to) 8 neighbors of board[row][col].

    Cells outside the board are treated as dead.
    """
    n_rows, n_cols = len(board), len(board[0])
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue  # skip the cell itself
            r, c = row + dr, col + dc
            if 0 <= r < n_rows and 0 <= c < n_cols:
                count += board[r][c]
    return count


def evolve(initial_state):
    """Return the next generation of a Game of Life board.

    The board is a list of lists of 0 (dead) and 1 (alive). The input is not
    modified; a new board is returned. Rules:
    - A live cell with 2 or 3 live neighbors survives.
    - A dead cell with exactly 3 live neighbors becomes alive.
    - Every other cell dies or stays dead.
    """
    if not initial_state or not initial_state[0]:
        return [row[:] for row in initial_state]

    next_state = []
    for row in range(len(initial_state)):
        next_row = []
        for col in range(len(initial_state[0])):
            neighbors = count_live_neighbors(initial_state, row, col)
            alive = initial_state[row][col] == 1
            if alive and neighbors in (2, 3):
                next_row.append(1)
            elif not alive and neighbors == 3:
                next_row.append(1)
            else:
                next_row.append(0)
        next_state.append(next_row)
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
