def evolve(initial_state):

    rows = len(initial_state)
    cols = len(initial_state[0])
    new_state = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            neighbors = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    # Skip the cell itself
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbors += initial_state[ni][nj]

            # Conway's rules
            if initial_state[i][j] == 1 and neighbors in (2, 3):
                new_state[i][j] = 1  # survival
            elif initial_state[i][j] == 0 and neighbors == 3:
                new_state[i][j] = 1  # birth
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
