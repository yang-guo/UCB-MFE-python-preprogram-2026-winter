
def evolve(initial_state):
    rows = len(initial_state)
    cols = len(initial_state[0])
    next_state = [[0 for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            neighbors = 0

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    nr = r + dr
                    nc = c + dc

                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors += initial_state[nr][nc]

            if initial_state[r][c] == 1:
                if neighbors in (2, 3):
                    next_state[r][c] = 1
            else:
                if neighbors == 3:
                    next_state[r][c] = 1

    return next_state

