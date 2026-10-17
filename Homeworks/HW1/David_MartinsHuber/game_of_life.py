def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0])

    final_state = [
        [0 for j in range(cols)]
        for i in range(rows)
    ]

    t = 0

    for i in range(rows):
        for j in range(cols):

            row_start = max(0, i - 1)
            row_end = min(rows, i + 2)

            col_start = max(0, j - 1)
            col_end = min(cols, j + 2)


            zone = [
                row[col_start:col_end]
                for row in initial_state[row_start:row_end]
            ]


            neighbors = sum(sum(row) for row in zone)
            neighbors -= initial_state[i][j] # substracting the middle cell

            value_cell_t0 = initial_state[i][j]

            if value_cell_t0 == 1:
                if neighbors == 2 or neighbors == 3:
                    value_cell_t1 = 1
                else:
                    value_cell_t1 = 0
            else:
                if neighbors == 3:
                    value_cell_t1 = 1
                else:
                    value_cell_t1 = 0

            final_state[i][j] = value_cell_t1
            t += 1

    assert t == rows * cols

    return final_state

test_case_1 = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
]

evolve(test_case_1)

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
