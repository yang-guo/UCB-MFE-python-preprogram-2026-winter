def evolve(initial_state):
    """Return the next generation of a Conway's Game of Life board.

    Each cell is either alive (1) or dead (0). Its next state depends
    on the number of live cells among its eight neighboring positions:
    - A live cell with fewer than 2 live neighbors dies.
    - A live cell with 2 or 3 live neighbors survives.
    - A live cell with more than 3 live neighbors dies.
    - A dead cell with exactly 3 live neighbors becomes alive.
    
    Args:
        initial_state: A 2D list representing the current board,
            where 1 denotes a live cell and 0 denotes a dead cell.

    Returns:
        A new 2D list representing the next generation of the board.
    """
    rows = len(initial_state)
    cols = len(initial_state[0])

    # Create a new board without modifying the original.
    next_state = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            neighbors = 0

            # Count live cells in the eight neighboring positions.
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue

                    ni = i + di
                    nj = j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbors += initial_state[ni][nj]

            alive = initial_state[i][j] == 1

            if alive:
                next_state[i][j] = int(neighbors in (2, 3))
            else:
                next_state[i][j] = int(neighbors == 3)

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
