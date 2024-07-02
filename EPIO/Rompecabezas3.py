from ortools.sat.python import cp_model

# Creates the model.
model = cp_model.CpModel()

#
# data
#
n = 5
nvecinos = 8
#[1, 1, 2, 2, 3, 3, 3, 3, 5, 5]
at = [0, 2, 2, 4, 0, 2, 0, 0, 0]

# neighborhood
Nij = {}
for i in range(n):
  for j in range(n):
    vecinos = {(i-1, j-1),(i-1, j),(i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)}
    aux = []
    for (k, l) in vecinos:
      if k >= 0 and l >= 0 and k < n and l < n:
          aux.append((k,l))
    Nij[(i,j)] = aux

#
# declare variables
#

literals={}
for i in range(n):
  for j in range(n):
    for t in range(nvecinos+1):
      literals[i, j, t] = model.NewIntVar(0, 1, "x[%i,%i,%i]" % (i, j, t))

#
# constraints
#

# make sure that each cell of the grid is assigned at most one node.
[model.Add(sum([literals[i, j, t] for t in range(1, nvecinos+1)]) <= 1) for i in range(n) for j in range(n)]

# the required number of nodes with t neighbors is taken into account.
[model.Add(sum([literals[i, j, t] for i in range(n) for j in range(n)]) == at[t]) for t in range(1, nvecinos+1)]

# a node assigned to a cell has at least t neighbors, but no restriction is imposed if no node is assigned to the cell.
[model.Add(sum([literals[k, l, t] for (k, l) in Nij[(i,j)] for t in range(1, nvecinos+1)]) >= sum([t * literals[i, j, t] for t in range(1, nvecinos+1)])) for i in range(n) for j in range(n)]

# implies that a node assigned to a cell has at most t neighbors, but no restriction is made if no node is assigned to the cell
[model.Add(sum([literals[k, l, t] for (k, l) in Nij[(i,j)] for t in range(1, nvecinos+1)]) <= sum([t * literals[i, j, t] for t in range(1, nvecinos+1)]) + 
           100 * (1 - sum([literals[i, j, t] for t in range(1, nvecinos+1)]))) for i in range(n) for j in range(n)]

#
# search and result
#

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)
#print(status == cp_model.OPTIMAL)

sol = []
for t in range(1, nvecinos+1):
  for i in range(n):
    for j in range(n):
      if solver.Value(literals[i, j, t]) == 1:
        sol.append((i,j))

for i in range(n):
  for j in range(n):
    if (i,j) in sol:        
      print('x', end=' ')
    else:
      print('-', end=' ')
  print()
print()