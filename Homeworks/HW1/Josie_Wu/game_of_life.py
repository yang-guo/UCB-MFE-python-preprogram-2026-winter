def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0])
    next_state = []

    for row in range(rows):
        new_row = []
        for col in range(cols):
            neighbors = 0

            # Count live cells in the eight surrounding cells.
            for r in range(max(0, row - 1), min(rows, row + 2)):
                for c in range(max(0, col - 1), min(cols, col + 2)):
                    if (r, c) != (row, col):
                        neighbors += initial_state[r][c]

            alive = initial_state[row][col]

            # Apply the Game of Life rules to determine the next state.
            if alive == 1 and neighbors in (2, 3):
                new_row.append(1)
            elif alive == 0 and neighbors == 3:
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
