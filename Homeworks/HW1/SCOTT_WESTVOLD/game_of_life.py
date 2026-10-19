import pandas as pd

"""
`get_neighbors(i = int, j = int, initial_state = list)` 
    Please Note Diagonal Neighbors (8 Comparisons Per Square)
return list
"""
def get_neighbors(i, j, initial_state):
    if (i == 0):
        nAbove = 0
    else:
        nAbove = initial_state.iloc[i - 1, j]
    
    if (i == (initial_state.shape[0] - 1)):
        nBelow = 0
    else:
        nBelow = initial_state.iloc[i + 1, j]
    
    if (j == 0):
        nLeft = 0
    else:
        nLeft =initial_state.iloc[i, j - 1]
    
    if (j == (initial_state.shape[1] - 1)):
        nRight = 0
    else:
        nRight = initial_state.iloc[i, j + 1]
        
    if ((i == 0) | (j == (initial_state.shape[1] - 1))):
        nTopRight = 0
    else:
        nTopRight = initial_state.iloc[i - 1, j + 1]
    
    if ((i == 0) | (j == 0)):
        nTopLeft = 0
    else:
        nTopLeft = initial_state.iloc[i - 1, j - 1]   
    
    if (i == (initial_state.shape[0] - 1)) | (j == (initial_state.shape[1] - 1)):
        nBottomRight = 0
    else:
        nBottomRight = initial_state.iloc[i + 1, j + 1]
    
    if ((i == (initial_state.shape[0] - 1)) | (j == 0)):
        nBottomLeft = 0
    else:
        nBottomLeft = initial_state.iloc[i + 1, j - 1]   
    
    nLive = nAbove + nBelow + nLeft + nRight + nTopRight + nTopLeft + nBottomRight + nBottomLeft
    return nLive

"""
`clean_updated_state(updated_state = pd.DataFrame)` 
    check first and last column. If all not alive, then remove.
    check first and last row. If all not alive, then remove.
    will not shrink grid size in evolve as we add a rows and columns at the end of each dimension at the start of the function.
return list
"""
def clean_updated_state(updated_state):
    cleaned_updated_state = updated_state.copy()
    totalLive = updated_state.iloc[0, :].sum()
    if totalLive == 0:
        cleaned_updated_state = cleaned_updated_state.iloc[1:, :]
    
    totalLive = updated_state.iloc[-1, :].sum()
    if totalLive == 0:
        cleaned_updated_state = cleaned_updated_state.iloc[:-1, :]
    
    totalLive = updated_state.iloc[:, 0].sum()
    if totalLive == 0:
        cleaned_updated_state = cleaned_updated_state.iloc[:, 1:]
    
    totalLive = updated_state.iloc[:, -1].sum()
    if totalLive == 0:
        cleaned_updated_state = cleaned_updated_state.iloc[:, :-1]
    
    return cleaned_updated_state



"""
`get_neighbors(i = int, j = int, initial_state = list)` 
    Conway Game of Life update function
    Add to gridsize at each bound to handle infinite grid
    Calculate number of neighbors including diagonal
    Implement Game of Life Update State Rules:
        Any live cell with fewer than two live neighbours dies, as if by underpopulation.
        Any live cell with two or three live neighbours lives on to the next generation.
        Any live cell with more than three live neighbours dies, as if by overpopulation.
        Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.[3][4]: 3get neighbors
    return Updated list of lists representing new grid.
    This function is built to handle an infinite grid so that if a neighbor on the edge of the grid becomes alive it will expand.
return list
"""
def evolve(initial_state):
    initial_state = pd.DataFrame(initial_state)


    # Infinite Grid. Add Row and Column to Beginning and End in case of new live cells increasing grid size
    zeros_row = initial_state.iloc[[0]] * 0
    initial_state = pd.concat([zeros_row, initial_state, zeros_row], axis = 0, ignore_index=True)
    zeros_col = initial_state[0]
    initial_state = pd.concat([zeros_col, initial_state, zeros_col], axis = 1, ignore_index=True)
    
    # Calculate Number of Neighbors
    n_neighbors = initial_state.copy() * 0
    for j in initial_state.columns:
        for i in initial_state.index:
            n_neighbors.iloc[i, j] = get_neighbors(i, j, initial_state)  

    # Update State
    updated_state = initial_state.copy()  * 0
    for j in initial_state.columns:
        for i in initial_state.index:
            if initial_state.iloc[i, j] == 1:
                if n_neighbors.iloc[i, j] < 2:
                    updated_state.iloc[i, j] = 0
                elif n_neighbors.iloc[i, j] <= 3:
                    updated_state.iloc[i, j] = 1
                elif n_neighbors.iloc[i, j] > 3:
                    updated_state.iloc[i, j] = 0
                else:
                    updated_state.iloc[i, j] = 0 # I know this is redundant but the complete logic is nice for reading the code.
            else:
                if n_neighbors.iloc[i, j] == 3:
                    updated_state.iloc[i, j] = 1
                else:
                    updated_state.iloc[i, j] = 0  # I know this is redundant but the complete logic is nice for reading the code.
    
    # Clean Extra Rows and Columns
    updated_state = clean_updated_state(updated_state)
    updated_state = updated_state.values.tolist()
    return updated_state
    
    pass


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
