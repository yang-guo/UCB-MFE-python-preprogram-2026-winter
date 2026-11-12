def evolve(initial_state):
    n = len(initial_state)
    m = len(initial_state[0])
    new_state = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            vive = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    r, c = i + di, j + dj
                    if 0 <= r < n and 0 <= c < m:
                        vive += initial_state[r][c]
            if initial_state[i][j] == 0:
                if vive == 3:
                    new_state[i][j] = 1
                else:
                    new_state[i][j] = 0
            else:
                if vive == 2 or vive == 3:
                    new_state[i][j] = 1
                else:
                    new_state[i][j] = 0

    return new_state

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
