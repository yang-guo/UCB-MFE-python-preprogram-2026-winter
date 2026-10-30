def evolve(initial_state: list[list[int]]) -> list[list[int]]:
    """Return the next generation of a Conway's Game of Life board.

    The board is a list of rows; each cell is 1 (alive) or 0 (dead). Cells outside
    the board count as dead (no wrap-around). The input board is not modified.

    Rules, applied to every cell at the same time:
    - a live cell with 2 or 3 live neighbours stays alive;
    - a dead cell with exactly 3 live neighbours becomes alive;
    - every other cell is dead in the next generation.
    """
    rows = len(initial_state)
    cols = len(initial_state[0]) if rows else 0

    def live_neighbours(r: int, c: int) -> int:
        # Sum the 3x3 block around (r, c), clipped to the board, minus the cell itself.
        return sum(
            initial_state[i][j]
            for i in range(max(r - 1, 0), min(r + 2, rows))
            for j in range(max(c - 1, 0), min(c + 2, cols))
            if (i, j) != (r, c)
        )

    def next_cell(r: int, c: int) -> int:
        n = live_neighbours(r, c)
        alive = initial_state[r][c] == 1
        return 1 if n == 3 or (alive and n == 2) else 0

    return [[next_cell(r, c) for c in range(cols)] for r in range(rows)]


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
print("Both test cases pass.")
