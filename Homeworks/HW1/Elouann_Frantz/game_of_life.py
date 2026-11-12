def evolve(initial_state):
    # Get grid dimensions
    rows = len(initial_state)
    cols = len(initial_state[0])
    
    # Create a new grid filled with 0s for the next state
    next_state = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Iterate through every cell in the grid
    for r in range(rows):
        for c in range(cols):
            # 1. Count live neighbors
            live_neighbors = 0
            
            # Check the 8 surrounding directions
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # Skip the cell itself
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    
                    # Ensure the neighbor is within grid boundaries
                    if 0 <= nr < rows and 0 <= nc < cols:
                        live_neighbors += initial_state[nr][nc]
            
            # 2. Apply Game of Life rules
            if initial_state[r][c] == 1:
                # Survival: 2 or 3 live neighbors
                if live_neighbors == 2 or live_neighbors == 3:
                    next_state[r][c] = 1
            else:
                # Reproduction: exactly 3 live neighbors
                if live_neighbors == 3:
                    next_state[r][c] = 1
                    
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
