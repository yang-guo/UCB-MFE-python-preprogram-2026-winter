def evolve(initial_state):
    """Return the next generation of a Game of Life board.

    Cells outside the board are treated as dead. The input is not modified.
    """
    n_rows = len(initial_state)
    n_cols = len(initial_state[0]) if n_rows else 0
    next_state = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            neighbors = sum(
                initial_state[r][c]
                for r in range(max(i - 1, 0), min(i + 2, n_rows))
                for c in range(max(j - 1, 0), min(j + 2, n_cols))
                if (r, c) != (i, j)
            )
            alive = initial_state[i][j] == 1
            row.append(1 if neighbors == 3 or (alive and neighbors == 2) else 0)
        next_state.append(row)
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
