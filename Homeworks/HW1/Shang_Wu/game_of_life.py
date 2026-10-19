def count_live_neighbors(board, row, col):
    """Count live cells among the 8 neighbors; cells off the board count as dead."""
    rows, cols = len(board), len(board[0])
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                count += board[r][c]
    return count


def evolve(initial_state):
    """Return the next generation of a Game of Life board without mutating the input."""
    if not initial_state or not initial_state[0]:
        return [row[:] for row in initial_state]

    next_state = []
    for r, row in enumerate(initial_state):
        next_row = []
        for c, cell in enumerate(row):
            neighbors = count_live_neighbors(initial_state, r, c)
            # A live cell survives with 2 or 3 neighbors; a dead cell is born with exactly 3.
            alive = neighbors == 3 or (cell == 1 and neighbors == 2)
            next_row.append(1 if alive else 0)
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
