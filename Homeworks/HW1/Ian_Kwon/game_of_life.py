def evolve(initial_state):
    # fill this out
    rows = len(initial_state)
    cols = len(initial_state[0])

    # Create an empty next-generation board
    next_state = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):

            # Count living neighbors
            neighbors = 0

            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:

                    # Exclude the cell itself
                    if di == 0 and dj == 0:
                        continue

                    ni = i + di
                    nj = j + dj

                    # Check board boundaries
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbors += initial_state[ni][nj]

            # Apply Game of Life rules
            if initial_state[i][j] == 1:
                if neighbors == 2 or neighbors == 3:
                    next_state[i][j] = 1

            else:
                if neighbors == 3:
                    next_state[i][j] = 1

    return next_state
    pass


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
