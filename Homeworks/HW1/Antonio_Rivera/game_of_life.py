def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0])
    new_state = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            # 1. Count the live neighbors of cell (r, c)
            n = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue  # skip the cell itself
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        n += initial_state[nr][nc]

            # 2. Apply the rules
            if initial_state[r][c] == 1 and n in (2, 3):
                new_state[r][c] = 1  # survives
            elif initial_state[r][c] == 0 and n == 3:
                new_state[r][c] = 1  # born

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
