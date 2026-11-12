def evolve(initial_state):
    #create new grid of all zeros
    new_grid = []
    for row in initial_state:
        new_row = []
        for item in row:
            new_row.append(0)
        new_grid.append(new_row)

    #function to count alive neighbors, 
    def count_neighbors(row, column):
        count = 0

        for row_change in [-1,0,1]:
            for column_change in [-1,0,1]:

                if row_change == 0 and column_change == 0:# skips current position
                    continue

                new_row = row + row_change
                new_column = column + column_change

                #skips if off board
                if new_row <0 or new_column <0 or new_row >=len(initial_state) or new_column >=len(initial_state[0]):
                    continue

                #add to count if neighbor is alive
                if initial_state[new_row][new_column]==1:
                    count = count + 1
        return count

    #loop through original_state and count neighbors, updating new_grid appropriately 
    for row in range(len(initial_state)):
        for column in range(len(initial_state[0])):

            neighbors = count_neighbors(row, column)

            if initial_state[row][column] == 1:
                # live cell survives with 2 or 3 live neighbors
                if neighbors == 2 or neighbors == 3:
                    new_grid[row][column] = 1
            else:
                # dead cell becomes alive with exactly 3 live neighbors
                if neighbors == 3:
                    new_grid[row][column] = 1
    


    return new_grid




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
