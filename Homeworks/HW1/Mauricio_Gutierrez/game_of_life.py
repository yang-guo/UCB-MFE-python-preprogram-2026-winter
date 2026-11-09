def evolve(initial_state):
    rows = len(initial_state)
    columns = len(initial_state[0])

    next_state = []
    for row in range(rows):
        next_state.append([0] * columns)
	
    for row in range(rows):
        for column in range(columns):
            living_neighbors = 0

            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    # Do not count the cell itself.
                    if row_offset == 0 and column_offset == 0:
                        continue

                    neighbor_row = row + row_offset
                    neighbor_column = column + column_offset

                    # Count only cells inside the board.
                    if (
                        0 <= neighbor_row < rows
                        and 0 <= neighbor_column < columns
                    ):
                        living_neighbors += initial_state[
                            neighbor_row
                        ][neighbor_column]

            if initial_state[row][column] == 1:
                if living_neighbors in (2, 3):
                    next_state[row][column] = 1
            elif living_neighbors == 3:
                next_state[row][column] = 1

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
