def evolve(initial_state):
    """Return one new generation without changing the original board.

    The board is rectangular, with 1 for alive and 0 for dead. Cells
    outside the supplied board are treated as dead.
    """
    row_count = len(initial_state)
    if row_count == 0:
        return []
    column_count = len(initial_state[0])

    # Use a separate board so all cells change at the same time.
    next_state = [
        [0 for _ in range(column_count)]
        for _ in range(row_count)
    ]

    for row in range(row_count):
        for column in range(column_count):
            live_neighbors = 0

            # Check the eight positions around the current cell.
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue  # Do not count the cell itself.

                    neighbor_row = row + row_offset
                    neighbor_column = column + column_offset
                    if (
                        0 <= neighbor_row < row_count
                        and 0 <= neighbor_column < column_count
                    ):
                        live_neighbors += initial_state[neighbor_row][neighbor_column]

            if initial_state[row][column] == 1:
                if live_neighbors in (2, 3):
                    next_state[row][column] = 1
            elif live_neighbors == 3:
                next_state[row][column] = 1
            # All other cells stay 0 on the new board.

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
