"""One generation of Conway's Game of Life on a finite 0/1 board."""


def evolve(board):
    """Return the next board. Cells outside the board count as dead."""
    rows = len(board)
    cols = len(board[0]) if rows else 0
    next_board = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            live_neighbors = 0
            for near_row in range(max(0, row - 1), min(rows, row + 2)):
                for near_col in range(max(0, col - 1), min(cols, col + 2)):
                    if (near_row, near_col) != (row, col):
                        live_neighbors += board[near_row][near_col]

            if live_neighbors == 3 or (board[row][col] and live_neighbors == 2):
                next_board[row][col] = 1

    return next_board


if __name__ == "__main__":
    # Independent checks. The course starter's two tests were not supplied.
    block = [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
    assert evolve(block) == block
    horizontal = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    vertical = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert evolve(horizontal) == vertical
    assert evolve(vertical) == horizontal
    print("Independent checks passed.")
