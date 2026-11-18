def evolve(initial_state):
    """Return the next generation, treating cells outside the board as dead."""
    rows = len(initial_state)
    cols = len(initial_state[0]) if rows else 0
    next_state = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            neighbors = 0
            for near_row in range(max(0, row - 1), min(rows, row + 2)):
                for near_col in range(max(0, col - 1), min(cols, col + 2)):
                    if near_row != row or near_col != col:
                        neighbors += initial_state[near_row][near_col]

            if neighbors == 3 or (initial_state[row][col] == 1 and neighbors == 2):
                next_state[row][col] = 1

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
