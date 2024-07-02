from ortools.sat.python import cp_model

# Creates the model.
model = cp_model.CpModel()

#
# data
#
n = 5
islas=[[(0,0), (0,1), (1,0), (1,1), (1,2), (2,0), (2,1), (3,0), (4,0)],
       [(0,2),(0,3),(0,4)],
       [(1,3),(1,4),(2,3),(2,4)],
       [(2,2),(3,1),(3,2),(4,1)],
       [(2,4),(3,4),(4,4),(4,2),(4,3)]]

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
[model.Add(sum([cuadricula[i][j] for j in range(n)]) == 1) for i in range(n)]

# exacly one assignment per column, all rows must be assigned
[model.Add(sum([cuadricula[i][j] for i in range(n)]) == 1) for j in range(n)]

# exacly one assignment per island
[model.Add(sum([cuadricula[islas[k][j][0]][islas[k][j][1]] for j in range(0, len(islas[k]))]) == 1) for k in range(n)]

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
