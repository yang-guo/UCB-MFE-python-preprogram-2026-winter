
def find_neighbors(row, col, initial_state, rows, cols):
  neighbors = 0
  for i in range(-1, 2):
    for j in range(-1, 2):
      # Looks at all cells in the vicinity of [i,j]
      if i == 0 and j == 0:
        # That's the current cell, ignore
        continue
      elif 0<= row+i < rows and 0 <= col+ j < cols:   
        # Look only at in-range cells
        if initial_state[row + i][col + j] == 1:
          # Cell is in-range and value is 1, so increment num neighbors
          neighbors += 1
  return neighbors

def evolve(initial_state):
  initial = initial_state
  rows = len(initial_state)
  cols = len(initial_state[0])
  next_state = [[0 for i in range(cols)] for j in range(rows)]
  for i in range(rows):
    for j in range(cols):
      neighbors = 0
      neighbors = find_neighbors(i, j, initial_state, rows, cols)
      if neighbors < 2:
        # Underpopulated -> DIE
        next_state[i][j] = 0
      elif neighbors > 3:
        # Overpopulated -> DIE
        next_state[i][j] = 0
      elif neighbors == 3:
        # Reproduce
        next_state[i][j] = 1
      else:
        next_state[i][j] = initial_state[i][j]
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


print("Begin")
assert evolve(test_case_1) == test_case_1
assert evolve(test_case_2) == test_case_2_next
print("Success")
