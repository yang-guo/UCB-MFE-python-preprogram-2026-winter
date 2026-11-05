def isvalid(row, col, n_rows, m_cols):
    # check cell validity within the matrix
    return row >= 0 and row < n_rows and col >= 0 and col < m_cols

def neighbours(row, col, state):
    # count number of alive neighbours for a cell
    res = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            new_row = row + dx
            new_col = col + dy
            if isvalid(new_row, new_col, len(state), len(state[0])):
                res += state[new_row][new_col]
    return res


def evolve(initial_state):
    # fill this out
    # game of life state iterator (n_states = 1)
    n_rows, m_cols = len(initial_state), len(initial_state[0])
    n_states = 1
    cur_state = initial_state
    for stage in range(1, n_states + 1):
        next_state = [[0 for _ in range(m_cols)] for _ in range(n_rows)]
        # assert(len(next_state) == n_rows and len(next_state[0]) == m_cols)
        for row in range(n_rows):
            for col in range(m_cols):
                alive_neighbours = neighbours(row, col, cur_state)
                if cur_state[row][col] == 1:
                    if alive_neighbours < 2 or alive_neighbours > 3:
                        next_state[row][col] = 0
                    else:
                        next_state[row][col] = 1
                else:
                    if alive_neighbours == 3:
                        next_state[row][col] = 1
        cur_state = next_state
    return cur_state


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
