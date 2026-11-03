def evolve(initial_state):
    result = [row[:] for row in initial_state]
    
    for row in range(len(initial_state)):
        for cell in range(len(initial_state[row])):
            temp_sum = 0
            for row_change in range(-1,2):
                for cell_change in range(-1,2):
                    if row_change == 0 and cell_change == 0:
                        continue
                    if row + row_change < 0 or row + row_change >= len(initial_state):
                        continue
                    if cell + cell_change < 0 or cell + cell_change >= len(initial_state[row]):
                        continue
                    temp_sum += initial_state[row + row_change][cell + cell_change]



            if initial_state[row][cell] == 1:
                if temp_sum < 2 or temp_sum > 3:
                    result[row][cell] = 0
            if initial_state[row][cell] == 0:
                if temp_sum == 3:
                    result[row][cell] = 1

    return result


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
