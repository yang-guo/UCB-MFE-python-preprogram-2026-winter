def evolve(initial_state):
    rows = len(initial_state)
    if rows == 0:
        return []
    
    cols = len(initial_state[0])

    new_state = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            living_neighbors = 0

            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        living_neighbors += initial_state[ni][nj]
            
            if living_neighbors == 3:
                new_state[i][j] = 1
            elif living_neighbors == 2 and initial_state[i][j] == 1:
                new_state[i][j] = initial_state[i][j]

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
