def evolve(initial_state):
    n = len(initial_state)
    m = len(initial_state[0])
    res = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            cnt = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    x = i + di
                    y = j + dj
                    if 0 <= x < n and 0 <= y < m:
                        cnt += initial_state[x][y]
            if initial_state[i][j] == 1 and (cnt == 2 or cnt == 3):
                res[i][j] = 1
            elif initial_state[i][j] == 0 and cnt == 3:
                res[i][j] = 1
    return res


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
