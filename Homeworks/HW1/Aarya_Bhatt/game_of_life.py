import copy

def evolve(board: list[list[int]]) -> list[list[int]]:
    """
    Evolves the Conway's Game of Life board by one generation.
    
    Rules:
    1. Any live cell with fewer than two live neighbours dies (underpopulation).
    2. Any live cell with two or three live neighbours lives on to the next generation.
    3. Any live cell with more than three live neighbours dies (overpopulation).
    4. Any dead cell with exactly three live neighbours becomes a live cell (reproduction).
    """
    if not board or not board[0]:
        return []

    rows = len(board)
    cols = len(board[0])
    new_board = [[0 for _ in range(cols)] for _ in range(rows)]
    
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for r in range(rows):
        for c in range(cols):
            live_neighbors = 0
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    live_neighbors += board[nr][nc]
            
            if board[r][c] == 1:
                if live_neighbors in (2, 3):
                    new_board[r][c] = 1
            else:
                if live_neighbors == 3:
                    new_board[r][c] = 1
                    
    return new_board


def test_evolve():
    # Test 1: Blinker oscillator
    blinker = [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ]
    expected_blinker = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]
    assert evolve(blinker) == expected_blinker, "Blinker test failed"

    # Test 2: Block still life
    block = [
        [1, 1],
        [1, 1]
    ]
    assert evolve(block) == block, "Block test failed"
    print("All tests passed successfully!")


if __name__ == "__main__":
    test_evolve()
