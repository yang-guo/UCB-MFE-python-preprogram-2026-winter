def evolve(initial_state):
    rows = len(initial_state)
    if rows == 0:
        return []

    cols = len(initial_state[0])

    # Build a new grid filled with zeros (all cells dead)
    next_state = []
    for r in range(rows):
        next_state.append([0] * cols)

    # Visit every cell
    for r in range(rows):
        for c in range(cols):
            neighbors = 0

            # Count live neighbors (cells outside the board count as dead)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    nr = r + dr
                    nc = c + dc

                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors += initial_state[nr][nc]

            # Apply the Game of Life rules
            if initial_state[r][c] == 1:
                if neighbors == 2 or neighbors == 3:
                    next_state[r][c] = 1
            elif neighbors == 3:
                next_state[r][c] = 1

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
