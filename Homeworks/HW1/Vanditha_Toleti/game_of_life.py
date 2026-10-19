def evolve(initial_state):
    # fill this out
    rows = len(initial_state)
    cols = len(initial_state[0])
    next_state = [[0]* cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            live_neighbors = 0
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di==0 and dj==0:
                        continue
                    new_i = i+di
                    new_j = j+dj
                    if (new_i>=0 and new_i<rows) and (new_j>=0 and new_j<cols):
                        live_neighbors += initial_state[new_i][new_j]
            if initial_state[i][j]==1:
                if live_neighbors in (2,3):
                    next_state[i][j]=1
            else:
                if live_neighbors == 3:
                    next_state[i][j]=1
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

print("All good!")
