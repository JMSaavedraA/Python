from ortools.sat.python import cp_model

# Creates the model.
model = cp_model.CpModel()

#
# data
#
n = 9
islas=[[(0,0), (1,0), (0,1), (0,2), (1,2), (0,3), (0,4), (1,4),(0,5),(1,5),(2,5),(1,6),(2,6),(3,6),(2,7),(3,7)],
       [(2,0),(3,0),(1,1),(2,1)],
       [(2,2),(1,3),(2,3),(3,3),(4,3),(5,3),(2,4),(3,4),(4,4),(5,4),(6,4),(3,5),(4,5),(5,5),(5,6)],
       [(0,6),(0,7),(1,7),(0,8),(1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(8,8)],
       [(4,0),(5,0),(3,1),(4,1),(3,2)],
       [(6,0),(7,0),(8,0),(5,1),(6,1),(7,1),(8,1),(4,2),(5,2),(6,2),(7,2),(8,2),(6,3)],
       [(4,6),(4,7),(5,7),(6,7)],
       [(7,3),(8,3),(7,4),(8,4),(6,5),(7,5),(6,6),(7,6)],
       [(8,5),(8,6),(7,7),(8,7)]]

# declare variables
cuadricula = []
for i in range(n):
  filas = []
  for j in range(n):
    filas.append(model.NewIntVar(0, 1, "x[%i,%i]" % (i, j)))
  cuadricula.append(filas)

#
# constraints
#

# exacly one assignment per row, all rows must be assigned
[model.Add(sum([cuadricula[i][j] for j in range(n)]) == 2) for i in range(n)]

# exacly one assignment per column, all rows must be assigned
[model.Add(sum([cuadricula[i][j] for i in range(n)]) == 2) for j in range(n)]

# exacly one assignment per island
[model.Add(sum([cuadricula[islas[k][j][0]][islas[k][j][1]] for j in range(0, len(islas[k]))]) == 2) for k in range(n)]

# 
[model.Add(cuadricula[i][j] + cuadricula[i+1][j] + cuadricula[i][j+1] + cuadricula[i+1][j+1] <= 1) for i in range(n-1) for j in range(n-1)]

#
# search and result
#

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)
#print(status == cp_model.OPTIMAL)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
  for i in range(n):
    for j in range(n):
      if solver.Value(cuadricula[i][j]) == 1:
        # There is a start in column j, row i.
        print('*', end=' ')
      else:
        print('-', end=' ')
    print()
  print()
else:
  print('No solution found.')
