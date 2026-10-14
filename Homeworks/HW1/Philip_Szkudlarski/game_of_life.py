def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0])
    
    next_state = []

    for i in range(rows):
        new_row = []
        
        for j in range(cols):
            neighbors = 0
            
            # Count the live neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    
                    ni = i + di
                    nj = j + dj
                    
                    # Ignore positions outside the grid
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbors += initial_state[ni][nj]
            
            # Apply the Game of Life rules
            if initial_state[i][j] == 1:
                if neighbors == 2 or neighbors == 3:
                    new_row.append(1)
                else:
                    new_row.append(0)
            else:
                if neighbors == 3:
                    new_row.append(1)
                else:
                    new_row.append(0)
        
        next_state.append(new_row)
    
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
