def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0]) if rows else 0
    next_state = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            # Count live neighbors among the 8 surrounding cells, skipping
            # positions outside the board (no wrap-around).
            neighbors = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbors += initial_state[ni][nj]

            alive = initial_state[i][j] == 1
            if alive and neighbors in (2, 3):
                next_state[i][j] = 1
            elif not alive and neighbors == 3:
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
