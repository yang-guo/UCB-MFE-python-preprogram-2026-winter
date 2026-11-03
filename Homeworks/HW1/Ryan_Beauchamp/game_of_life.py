def count_neighbors(board, r, c):
    rows = len(board) # number of rows
    cols = len(board[0]) # number of columns
    count = 0
    for row_offset in (-1, 0,1): # iterate through and check surrounding cells
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0: 
                continue  # You do not want to count the cell itself, only the 8 surrounding neighbors
            row_check = r + row_offset
            col_check = c + col_offset
            if 0 <= row_check < rows and 0 <= col_check < cols: # make sure to handle the edges
                count += board[row_check][col_check]
    return count

def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0])
    new_state = [[0] * cols for _ in range(rows)] # Create new empty matrix of the same size as the one being passed through the function
    for r in range(rows):
        for c in range(cols): # Nested loops to itereate through each location of the matrix
            num_neighbors = count_neighbors(initial_state, r, c)
            if initial_state[r][c] == 1: # check if the cell is alive or not
               if num_neighbors == 2 or num_neighbors == 3: # Alive with 2 or 3 neighbors, then it gets to continue living
                    new_state[r][c] = 1
            else: # dead cells go here
                if num_neighbors == 3: # if the cell is dead with exactly 3 neighbors, it is revived
                    new_state[r][c] = 1
    return new_state


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
