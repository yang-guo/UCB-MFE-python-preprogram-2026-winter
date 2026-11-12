def evolve(initial_state):
    """Return the next generation of Conway's Game of Life."""
    if not initial_state:
        return []

    rows, cols = len(initial_state), len(initial_state[0])

    padded = (
        [[0] * (cols + 2)]
        + [[0, *row, 0] for row in initial_state]
        + [[0] * (cols + 2)]
    )

    SURVIVE, BIRTH = {2, 3}, {3} 
    next_state = []

    for r in range(1, rows + 1):      
        new_row = []
        for c in range(1, cols + 1):
            n = sum(
                padded[r + dr][c + dc]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if dr or dc             
            )
            rule = SURVIVE if padded[r][c] else BIRTH
            new_row.append(1 if n in rule else 0)
        next_state.append(new_row)

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
