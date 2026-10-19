def evolve(initial_state):
    if not initial_state or not initial_state[0]:
        return [row[:] for row in initial_state]

    rows, cols = len(initial_state), len(initial_state[0])
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

    next_state = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            # Count live neighbours (cells outside the grid count as dead)
            live_neighbours = 0
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    live_neighbours += initial_state[nr][nc]

            if initial_state[r][c] == 1:
                # Rules 1-3: survives only with 2 or 3 live neighbours
                next_state[r][c] = 1 if live_neighbours in (2, 3) else 0
            else:
                # Rule 4: reproduction with exactly 3 live neighbours
                next_state[r][c] = 1 if live_neighbours == 3 else 0

    #print(next_state)
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
