def evolve(initial_state):
    # fill this out
    moves = [(-1,-1), (0,-1), (1,-1),
             (-1,0), (1,0),
             (-1,1), (0,1), (1,1)]
    mapping={3:0, 0:0, 1:1, 2:1} # 2=new dead, 3=new alive
    sum_map={0:2, 1:2,
             2:1, 3:1, 
             4:2, 5:2, 6:2, 7:2, 8:2}

    m=len(initial_state)
    n=len(initial_state[0])

    for i in range(m):
        for j in range(n):
            neighbor_sum = 0
            for x, y in moves:
                n_i, n_j=i+x, j+y

                # boundaries
                if 0<=n_i<m and 0<=n_j<n:
                    neighbor_sum+=mapping[initial_state[n_i][n_j]]

            if initial_state[i][j]==0 and neighbor_sum==3:
                initial_state[i][j]=3
            elif initial_state[i][j]==1:
                initial_state[i][j]=sum_map[neighbor_sum]

    for i in range(m):
        for j in range(n):
            initial_state[i][j] %= 2

    return(initial_state)


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
