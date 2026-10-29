def evolve(initial_state):
    # fill this out
    board_copy=[row.copy() for row in initial_state]
    moves = [(-1,-1), (0,-1), (1,-1),
             (-1,0), (1,0),
             (-1,1), (0,1), (1,1)]

    m=len(initial_state)
    n=len(initial_state[0])

    for i in range(m):
        for j in range(n):
            neighbor_sum = 0
            for x, y in moves:
                n_i, n_j=i+x, j+y

                if 0<=n_i<m and 0<=n_j<n:
                    neighbor_sum+=initial_state[n_i][n_j]

            if initial_state[i][j]==0 and neighbor_sum==3:
                board_copy[i][j]=1
            elif initial_state[i][j]==1 and (neighbor_sum==2 or neighbor_sum==3):
                board_copy[i][j]=1
            else:
                board_copy[i][j]=0

    return(board_copy)


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
