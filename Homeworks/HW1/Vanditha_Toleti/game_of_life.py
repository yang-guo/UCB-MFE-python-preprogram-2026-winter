def evolve(initial_state):
    # Get the dimensions of the board.
    rows = len(initial_state)
    cols = len(initial_state[0])
    # Create a new board filled with dead cells (0).
    next_state = [[0]* cols for _ in range(rows)]
    # Visit every cell in the original board.
    for i in range(rows):
        for j in range(cols):
            live_neighbors = 0
            # Check the eight neighboring positions around the current cell.
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    # Skip the current cell itself.
                    if di==0 and dj==0:
                        continue
                    # Calculate the coordinates of the neighboring cell.
                    new_i = i+di
                    new_j = j+dj
                    
                    # Count live neighbors only if they are within the board.
                    if (new_i>=0 and new_i<rows) and (new_j>=0 and new_j<cols):
                        live_neighbors += initial_state[new_i][new_j]
    
            # Apply Conway's Game of Life rules.
            if initial_state[i][j]==1:
                # A live cell survives if it has two or three live neighbors.
                if live_neighbors in (2,3):
                    next_state[i][j]=1
            else:
                # A dead cell becomes alive if it has exactly three live neighbors.
                if live_neighbors == 3:
                    next_state[i][j]=1
    # Return the next generation without modifying the original board.
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

print("All good!")
