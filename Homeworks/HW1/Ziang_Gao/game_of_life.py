def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0]) if rows > 0 else 0
    next_state = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            live_neighbors = 0
            for x in range(max(0, i - 1), min(rows, i + 2)):
                for y in range(max(0, j - 1), min(cols, j + 2)):
                    if (x, y) != (i, j):
                        live_neighbors += initial_state[x][y]
            if initial_state[i][j] == 1:
                next_state[i][j] = 1 if live_neighbors in [2, 3] else 0
            else:
                next_state[i][j] = 1 if live_neighbors == 3 else 0
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


#print("All tests passed")