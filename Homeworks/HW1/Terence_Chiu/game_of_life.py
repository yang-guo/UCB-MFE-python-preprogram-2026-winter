def evolve(initial_state):
    m, n = len(initial_state), len(initial_state[0])

    next_state = [[0] * n for _ in range(m)]

    for i in range(m):
        for j in range (n):
            cell = initial_state[i][j]
            neighbors = check_neighbor(initial_state, i, j)
            if cell == 1:
                if neighbors < 2:
                    cell = 0
                elif neighbors < 4:
                    cell = 1
                else:
                    cell = 0
            else:
                if neighbors == 3:
                    cell = 1
            next_state[i][j] = cell

    return next_state

def check_neighbor(board, row, col):
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < len(board) and 0 <= c < len(board[0]):
                count += board[r][c]
    return count



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
