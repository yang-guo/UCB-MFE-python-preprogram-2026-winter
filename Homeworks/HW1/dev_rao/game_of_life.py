def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0]) if rows else 0
    next_state = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            # count live neighbors ignore anything off the board
            neighbors = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors += initial_state[nr][nc]

            if initial_state[r][c] == 1:
                next_state[r][c] = 1 if neighbors in (2, 3) else 0
            else:
                next_state[r][c] = 1 if neighbors == 3 else 0

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
