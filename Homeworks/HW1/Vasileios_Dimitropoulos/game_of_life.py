def evolve(initial_state):
    # Get the size of the board (number of rows and columns)
    num_rows = len(initial_state)
    num_cols = len(initial_state[0])

    # Create a new board filled with zeros to hold the next generation
    next_state = [[0] * num_cols for _ in range(num_rows)]

    # Go through every cell on the board
    for row in range(num_rows):
        for col in range(num_cols):
            # Count how many of the 8 neighbouring cells are alive
            live_neighbors = 0
            for row_offset in [-1, 0, 1]:
                for col_offset in [-1, 0, 1]:
                    # Skip the cell itself
                    if row_offset == 0 and col_offset == 0:
                        continue
                    neighbor_row = row + row_offset
                    neighbor_col = col + col_offset
                    # Only count neighbours that are inside the board
                    if 0 <= neighbor_row < num_rows and 0 <= neighbor_col < num_cols:
                        live_neighbors += initial_state[neighbor_row][neighbor_col]

            # Apply Conway's rules
            if initial_state[row][col] == 1:
                # A live cell survives only with 2 or 3 live neighbours
                if live_neighbors == 2 or live_neighbors == 3:
                    next_state[row][col] = 1
            else:
                # A dead cell becomes alive with exactly 3 live neighbours
                if live_neighbors == 3:
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
print("All tests passed!")
