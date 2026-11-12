def evolve(initial_state):
    #  the dimensions of the grid
    rows = len(initial_state)
    cols = len(initial_state[0]) if rows > 0 else 0
    
    # initialize the next generation grid with all dead cells (0)
    next_state = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # iterate through every cell in the grid
    for i in range(rows):
        for j in range(cols):
            # Count the live neighbors around the current cell
            live_neighbors = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue # Skip the current cell
                    
                    ni, nj = i + di, j + dj
                    # the neighbor is within the grid boundaries
                    if 0 <= ni < rows and 0 <= nj < cols:
                        live_neighbors += initial_state[ni][nj]
            
            # Game of Life rules
            if initial_state[i][j] == 1: # if the cell is currently alive
                if live_neighbors == 2 or live_neighbors == 3:
                    next_state[i][j] = 1 # survives
            else: # if the cell is currently dead
                if live_neighbors == 3:
                    next_state[i][j] = 1 # becomes alive through reproduction
                
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
