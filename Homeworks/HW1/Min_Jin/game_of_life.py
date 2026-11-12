def evolve(initial_state):
    # Helper function for checking if any of the 8 neighbors is alive
    def count_live_neighbors(board, r, c):
        live_neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                # skip itself 
                if dr == 0 and dc == 0: continue

                # find neighbor position
                nr = r + dr
                nc = c + dc

                # skip if neighbor is offboard
                if nr < 0 or nr >= len(board) or nc < 0 or nc >= len(board[0]):
                    continue

                # find number of neighbors alive
                if board[nr][nc] == 1:
                    live_neighbors += 1
        return live_neighbors

    # get number of rows and cols of the board
    rows = len(initial_state)
    cols = len(initial_state[0])

    # new board 
    new_board = [[0] * cols for _ in range(rows)]

    # check living neighbors for each index and apply rules for live/dead
    for row in range(rows):
        for col in range(cols):
            live_neighbor_cnt = count_live_neighbors(initial_state, row, col)
            if initial_state[row][col] == 1 and live_neighbor_cnt in (2,3):
                new_board[row][col] = 1
            elif initial_state[row][col] == 0 and live_neighbor_cnt == 3:
                new_board[row][col] = 1

    return new_board





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
