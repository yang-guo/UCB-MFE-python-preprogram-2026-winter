def evolve(initial_state):
    """Return the next generation of a Game of Life board."""
    rows = len(initial_state)
    cols = len(initial_state[0])
    new_state = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            # We need to count the live neighbors of cell (r, c)
            live_neighbors = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        live_neighbors += initial_state[nr][nc]

            # Then, we apply the rules of the game
            if initial_state[r][c] == 1 and live_neighbors in (2, 3):
                new_state[r][c] = 1
            elif initial_state[r][c] == 0 and live_neighbors == 3:
                new_state[r][c] = 1

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
