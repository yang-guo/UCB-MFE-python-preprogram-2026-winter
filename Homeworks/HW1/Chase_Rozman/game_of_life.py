def evolve(board):
    if not board or not board[0]:
        return []
    
    rows = len(board)
    cols = len(board[0])
    
    # 1. Create the new board filled with zeros
    next_board = [[0] * cols for _ in range(rows)]
    
    # 2. The 8 directions around any cell (row offsets, col offsets)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # 3. Loop through every single cell
    for r in range(rows):
        for c in range(cols):
            
            # --- CALCULATE live_neighbors HERE ---
            live_neighbors = 0
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                
                # Check if the neighbor coordinate is actually inside the grid bounds
                if 0 <= nr < rows and 0 <= nc < cols:
                    if board[nr][nc] == 1:
                        live_neighbors += 1
            # ---------------------------------------
            
            # 4. Apply the rules using live_neighbors
            if board[r][c] == 1:
                if live_neighbors in (2, 3):
                    next_board[r][c] = 1  # Survives
                else:
                    next_board[r][c] = 0  # Dies
            else:
                if live_neighbors == 3:
                    next_board[r][c] = 1  # Reproduction
                else:
                    next_board[r][c] = 0  # Stays dead
                    
    return next_board

if __name__ == "__main__":
    # Test block
    test_board = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]
    result = evolve(test_board)
    print("Evolution result:", result)