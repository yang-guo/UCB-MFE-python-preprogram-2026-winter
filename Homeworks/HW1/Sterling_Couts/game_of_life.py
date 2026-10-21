def evolve(initial_state):
    """Return the next generation of a rectangular Game of Life board.

    Cells contain 1 for alive or 0 for dead.
    Positions outside the board are treated as dead.
    The original board is left unchanged.
    """
    rows = len(initial_state)
    columns = len(initial_state[0]) if rows else 0

    # Build a separate board so every update uses the original generation.
    # Each row must be its own list to allow independent cell updates.
    next_state = [[0] * columns for _ in range(rows)]

    for row in range(rows):
        for column in range(columns):
            live_neighbors = 0

            # Check the surrounding positions, including diagonals.
            # Offsets of -1, 0, and 1 mean before, same, and after.
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):

                    # A cell is not one of its own neighbors.
                    if row_offset == 0 and column_offset == 0:
                        continue

                    neighbor_row = row + row_offset
                    neighbor_column = column + column_offset

                    # Only count neighbors that are inside the board.
                    if (
                        0 <= neighbor_row < rows
                        and 0 <= neighbor_column < columns
                    ):
                        live_neighbors += initial_state[
                            neighbor_row
                        ][neighbor_column]

            # With three live neighbors, either a birth or survival occurs.
            if live_neighbors == 3:
                next_state[row][column] = 1

            # An already-living cell also survives with two live neighbors.
            elif initial_state[row][column] == 1 and live_neighbors == 2:
                next_state[row][column] = 1

            # All other cells remain dead (the board was initialized to 0).

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
