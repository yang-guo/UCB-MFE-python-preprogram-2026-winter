def count_live_neighbours(board, r, c):
    """Number of live cells among the 8 neighbours of (r, c); off-board counts as dead."""
    rows, cols = len(board), len(board[0])
    total = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue  # a cell is not its own neighbour
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                total += board[nr][nc]
    return total


def evolve(initial_state):
    """Return the next generation as a new board; the input is left unchanged."""
    rows, cols = len(initial_state), len(initial_state[0])
    next_state = [[0] * cols for _ in range(rows)]  # fresh rows, not [[0] * cols] * rows

    for r in range(rows):
        for c in range(cols):
            n = count_live_neighbours(initial_state, r, c)  # always read the OLD board
            if initial_state[r][c] == 1:
                next_state[r][c] = 1 if n in (2, 3) else 0
            else:
                next_state[r][c] = 1 if n == 3 else 0

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
