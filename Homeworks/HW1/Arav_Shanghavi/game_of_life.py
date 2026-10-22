def evolve(initial_state):
    height = len(initial_state)
    width = len(initial_state[0]) if height else 0
    successor = [[0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            alive_around = 0
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    if oy == 0 and ox == 0:
                        continue
                    ny, nx = y + oy, x + ox
                    if 0 <= ny < height and 0 <= nx < width:
                        alive_around += initial_state[ny][nx]

            if initial_state[y][x] == 1 and alive_around in (2, 3):
                successor[y][x] = 1
            elif initial_state[y][x] == 0 and alive_around == 3:
                successor[y][x] = 1

    return successor


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
