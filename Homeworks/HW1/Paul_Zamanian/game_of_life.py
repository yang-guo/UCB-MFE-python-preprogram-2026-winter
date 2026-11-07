def evolve(initial_state):
    rows, cols = len(initial_state), len(initial_state[0])
    next_state = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            # cells outside the board count as dead
            neighbours = sum(
                initial_state[r][c]
                for r in range(max(i - 1, 0), min(i + 2, rows))
                for c in range(max(j - 1, 0), min(j + 2, cols))
            ) - initial_state[i][j]
            if neighbours == 3 or (initial_state[i][j] == 1 and neighbours == 2):
                next_state[i][j] = 1
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
