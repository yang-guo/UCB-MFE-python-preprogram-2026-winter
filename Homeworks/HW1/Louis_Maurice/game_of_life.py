def neighbors(state, i, j):
    res = []
    pos = [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i+1, j+1), (i-1, j+1), (i+1, j-1)]
    for p in pos:
        a = p[0]
        b = p[1]
        if (a >= 0 and b >= 0) and (a < len(state) and b < len(state[0])) and state[a][b] == 1:
            res.append((a, b))
    return res

def evolve(initial_state):
    # fill this out
    n = len(initial_state)
    m = len(initial_state[0])
    new_state = []
    for i in range(n):
        tmp = []
        for j in range(m):
            neigh = neighbors(initial_state, i, j)
            if initial_state[i][j] == 0 and len(neigh) == 3:
                tmp.append(1)
            elif initial_state[i][j] == 1 and (len(neigh) < 2 or len(neigh) > 3):
                tmp.append(0)
            else:
                tmp.append(initial_state[i][j])
        new_state.append(tmp)
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
