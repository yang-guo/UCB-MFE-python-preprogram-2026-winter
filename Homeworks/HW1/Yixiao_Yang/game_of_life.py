def evolve(initial_state):
    """Return the next generation, treating cells outside the board as dead."""
    rows = len(initial_state)
    cols = len(initial_state[0]) if rows else 0
    next_state = [[0 for _ in range(cols)] for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            live_neighbors = 0

            # Count the eight surrounding cells in the original board.
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue

                    neighbor_row = row + row_offset
                    neighbor_col = col + col_offset
                    if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                        live_neighbors += initial_state[neighbor_row][neighbor_col]

            # Birth requires three neighbors; survival requires two or three.
            if live_neighbors == 3 or (
                initial_state[row][col] == 1 and live_neighbors == 2
            ):
                next_state[row][col] = 1

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
