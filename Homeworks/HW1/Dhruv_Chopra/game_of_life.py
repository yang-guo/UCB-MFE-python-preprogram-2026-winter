def evolve(initial_state):
    # Count rows and columns separately so non-square boards work too
    rows = len(initial_state)
    cols = len(initial_state[0]) if rows else 0

    # Build a new board so every cell is computed from the original state
    arr = [[0] * cols for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            # Count live neighbors in the 8 surrounding cells,
            # treating anything off the edge of the board as dead
            live_neighbors = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        live_neighbors += initial_state[ni][nj]

            # Live cell survives with 2 or 3 neighbors; dead cell is born with exactly 3
            if initial_state[i][j] == 1 and live_neighbors in (2, 3):
                arr[i][j] = 1
            elif initial_state[i][j] == 0 and live_neighbors == 3:
                arr[i][j] = 1
    return arr

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
