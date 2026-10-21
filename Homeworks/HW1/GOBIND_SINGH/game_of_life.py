def evolve(initial_state):
    neighbor_count = [[0 for _ in range(len(initial_state[0]))] for _ in range(len(initial_state))]

    for i in range(len(initial_state)):
        for j in range(len(initial_state[0])):
            live_count = 0
            if i > 0:
                live_count += initial_state[i-1][j]
            if i < len(initial_state) - 1:
                live_count += initial_state[i+1][j]
            if j > 0:
                live_count += initial_state[i][j-1]
            if j < len(initial_state[0]) - 1:
                live_count += initial_state[i][j+1]
            if i > 0 and j > 0:
                live_count += initial_state[i-1][j-1]
            if i > 0 and j < len(initial_state[0]) - 1:
                live_count += initial_state[i-1][j+1]
            if i < len(initial_state) - 1 and j > 0:
                live_count += initial_state[i+1][j-1]
            if i < len(initial_state) - 1 and j < len(initial_state[0]) - 1:
                live_count += initial_state[i+1][j+1]
            neighbor_count[i][j] = live_count

    for i in range(len(initial_state)):
        for j in range(len(initial_state[0])):
            if initial_state[i][j] == 1:
                if neighbor_count[i][j] < 2 or neighbor_count[i][j] > 3:
                    initial_state[i][j] = 0
            else:
                if neighbor_count[i][j] == 3:
                    initial_state[i][j] = 1
    return initial_state




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
