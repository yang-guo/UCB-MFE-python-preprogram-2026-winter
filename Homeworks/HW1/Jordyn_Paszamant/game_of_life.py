def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0])

    def count_neighbors(r, c):
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue  
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    count += initial_state[nr][nc]
        return count

    next_state = []
    for r in range(rows):
        new_row = []
        for c in range(cols):
            n = count_neighbors(r, c)
            if initial_state[r][c] == 1:
                new_row.append(1 if n in (2, 3) else 0)
            else:
                new_row.append(1 if n == 3 else 0)
        next_state.append(new_row)

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
print("All tests passed!")

