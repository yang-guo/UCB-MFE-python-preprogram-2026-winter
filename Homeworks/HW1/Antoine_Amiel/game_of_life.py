def evolve(initial_state):
    n_rows = len(initial_state)
    n_cols = len(initial_state[0])
    next_state = []

    for i in range(n_rows):
        new_row = []
        for j in range(n_cols):
            # 1. Count live neighbours (the 8 surrounding cells)
            live = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue  # skip the cell itself
                    r, c = i + di, j + dj
                    if 0 <= r < n_rows and 0 <= c < n_cols:  # stay inside the board
                        live += initial_state[r][c]

            # 2. Apply the rules
            if initial_state[i][j] == 1:
                new_row.append(1 if live in (2, 3) else 0)
            else:
                new_row.append(1 if live == 3 else 0)

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
