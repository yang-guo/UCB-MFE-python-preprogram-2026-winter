def evolve(initial_state):
    """Return the next board without modifying the original board."""
    if not initial_state:
        return []

    rows = len(initial_state)
    cols = len(initial_state[0])
    next_state = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            neighbors = 0

            # Count live cells in the eight neighboring positions.
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    nr = r + dr
                    nc = c + dc

                    # Positions outside the board are treated as dead.
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors += initial_state[nr][nc]

            # Birth: exactly three neighbors.
            # Survival: a live cell with two or three neighbors.
            next_state[r][c] = int(
                neighbors == 3
                or (initial_state[r][c] == 1 and neighbors == 2)
            )

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

print("All tests passed!")
