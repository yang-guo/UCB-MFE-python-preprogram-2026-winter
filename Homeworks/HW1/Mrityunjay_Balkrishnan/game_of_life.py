def evolve(initial_state):
    """Return the next generation of a finite Conway's Game of Life board.

    Cells outside the supplied board are treated as dead. The input board is
    not modified, and every cell is updated from the same original generation.
    """
    row_count = len(initial_state)
    if row_count == 0:
        return []

    column_count = len(initial_state[0])
    next_state = []

    for row in range(row_count):
        next_row = []
        for column in range(column_count):
            live_neighbors = 0

            for neighbor_row in range(max(0, row - 1), min(row_count, row + 2)):
                for neighbor_column in range(
                    max(0, column - 1), min(column_count, column + 2)
                ):
                    if (neighbor_row, neighbor_column) == (row, column):
                        continue
                    live_neighbors += initial_state[neighbor_row][neighbor_column]

            if initial_state[row][column] == 1:
                next_row.append(1 if live_neighbors in (2, 3) else 0)
            else:
                next_row.append(1 if live_neighbors == 3 else 0)

        next_state.append(next_row)

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
