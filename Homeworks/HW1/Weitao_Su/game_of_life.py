

def count_neighbors(x,y,state):
    res=0
    for i in [1,0,-1]:
        newx=x+i
        if newx<0 or newx>=len(state):
            continue
        for j in [1,0,-1]:
            if i==0 and j==0:
                continue
            newy=y+j
            if newy<0 or newy>=len(state[0]):
                continue
            if state[newx][newy]==1:
                res+=1
    return res

def evolve(initial_state):
    # fill this out
    # pass
    state = [[0]*len(initial_state[0]) for _ in range(len(initial_state))]
    for i in range(len(initial_state)):
        for j in range(len(initial_state[0])):
            cnt=count_neighbors(i,j,initial_state)
            if initial_state[i][j]==1:
                if cnt==2 or cnt==3:
                    state[i][j]=1
                else:
                    state[i][j]=0
            else:
                if cnt==3:
                    state[i][j]=1
                else:
                    state[i][j]=0
    return state




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
