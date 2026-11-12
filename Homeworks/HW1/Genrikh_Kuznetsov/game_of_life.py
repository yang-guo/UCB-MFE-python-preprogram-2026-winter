def evolve(initial_state):
    next_state = [row.copy() for row in initial_state]
    length = len(initial_state)

    for i in range(length):
        for j in range(length):
            neighbors = 0

            # Not on a vertical border
            if i > 0 and i < length - 1:

                # Not on a horizontal border
                if j > 0 and j < length - 1:
                    neighbors = neighbors + (initial_state[i-1][j-1] + initial_state[i-1][j] + initial_state[i-1][j+1]
                                             + initial_state[i][j-1] + initial_state[i][j+1]
                                             + initial_state[i+1][j-1] + initial_state[i+1][j] + initial_state[i+1][j+1]
                    )

                # On the right border
                elif j > 0:
                    neighbors = neighbors + (initial_state[i-1][j-1] + initial_state[i-1][j]
                                             + initial_state[i][j-1]
                                             + initial_state[i+1][j-1] + initial_state[i+1][j]
                    )

                # On the left border
                else:
                    neighbors = neighbors + (initial_state[i-1][j] + initial_state[i-1][j+1]
                                             + initial_state[i][j+1]
                                             + initial_state[i+1][j] + initial_state[i+1][j+1]
                    )

            # Lower border
            elif i > 0:

                # Not on a horizontal border
                if j > 0 and j < length - 1:
                    neighbors = neighbors + (initial_state[i-1][j-1] + initial_state[i-1][j] + initial_state[i-1][j+1] 
                                             + initial_state[i][j-1] + initial_state[i][j+1])
                
                # On the right border
                elif j > 0:
                    neighbors = neighbors + (initial_state[i-1][j-1] + initial_state[i-1][j]
                                             + initial_state[i][j-1]
                    )
                
                # On the left border
                else:
                    neighbors = neighbors + (initial_state[i-1][j] + initial_state[i-1][j+1]
                                             + initial_state[i][j+1]
                    )
            
            # Upper border
            else:
                # Not on a horizontal border
                if j > 0 and j < length - 1:
                    neighbors = neighbors + (
                                             + initial_state[i][j-1] + initial_state[i][j+1]
                                             + initial_state[i+1][j-1] + initial_state[i+1][j] + initial_state[i+1][j+1]
                    )
                
                # On the right border
                elif j > 0:
                    neighbors = neighbors + (
                                             + initial_state[i][j-1]
                                             + initial_state[i+1][j-1] + initial_state[i+1][j]
                    )
                
                # On the left border
                else:
                    neighbors = neighbors + (
                                             + initial_state[i][j+1]
                                             + initial_state[i+1][j] + initial_state[i+1][j+1]
                    )

            # Alive
            if initial_state[i][j] == 1:
                # Die
                if neighbors < 2 or neighbors > 3:
                    next_state[i][j] = 0
            # Dead
            else:
                # Live
                if neighbors == 3:
                    next_state[i][j] = 1

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
