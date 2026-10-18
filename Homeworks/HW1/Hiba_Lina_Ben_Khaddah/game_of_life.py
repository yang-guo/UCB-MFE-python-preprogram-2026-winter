def evolve(initial_state):
    """Return the next generation of a Conway's Game of Life board.

    The input is a rectangular list of lists where 1 represents a living
    cell and 0 represents a dead cell. The function returns a new board and
    does not modify the original board.
    """
    # Determine the dimensions of the board.
    rows = len(initial_state)

    # An empty board remains empty.
    if rows == 0:
        return []

    columns = len(initial_state[0])

    # Start with a new board containing only dead cells.
    next_state = [[0 for _ in range(columns)] for _ in range(rows)]

    # Visit each cell and calculate its next state from the original board.
    for row in range(rows):
        for column in range(columns):
            living_neighbors = 0

            # Check all eight positions surrounding the current cell.
            for row_change in (-1, 0, 1):
                for column_change in (-1, 0, 1):
                    # A cell is not its own neighbor.
                    if row_change == 0 and column_change == 0:
                        continue

                    neighbor_row = row + row_change
                    neighbor_column = column + column_change

                    # Positions outside the board are treated as dead.
                    if (
                        0 <= neighbor_row < rows
                        and 0 <= neighbor_column < columns
                    ):
                        living_neighbors += initial_state[neighbor_row][
                            neighbor_column
                        ]

            current_cell = initial_state[row][column]

            # A living cell survives with two or three living neighbors.
            if current_cell == 1 and living_neighbors in (2, 3):
                next_state[row][column] = 1

            # A dead cell becomes alive with exactly three living neighbors.
            elif current_cell == 0 and living_neighbors == 3:
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
