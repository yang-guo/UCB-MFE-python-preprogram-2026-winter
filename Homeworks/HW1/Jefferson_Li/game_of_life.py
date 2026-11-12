def evolve(initial_state):
    next_state = [[0] * len(initial_state[0]) for _ in range(len(initial_state))]
    for i in range(len(initial_state)):
        for j in range(len(initial_state[0])):
            cur = initial_state[i][j]
            cnt = count_neighbor(initial_state, i, j)
            if cur == 1:
                if cnt == 2 or cnt == 3:
                    next_state[i][j] = 1
            else:
                if cnt == 3:
                    next_state[i][j] = 1
    return next_state


def count_neighbor(state, row, col):
    height = len(state)
    width = len(state[0])
    cnt = 0
    if row > 0:
        cnt += state[row - 1][col]
        if col > 0:
            cnt += state[row - 1][col - 1]
        if col < width - 1:
            cnt += state[row - 1][col + 1]

    if row < height - 1:
        cnt += state[row + 1][col]
        if col > 0:
            cnt += state[row + 1][col - 1]
        if col < width - 1:
            cnt += state[row + 1][col + 1]

    if col > 0:
        cnt += state[row][col - 1]

    if col < width - 1:
        cnt += state[row][col + 1]

    return cnt

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
