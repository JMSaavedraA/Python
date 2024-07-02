from ortools.sat.python import cp_model

# Creates the model.
model = cp_model.CpModel()

#
# data
#
colores = 4
nodos = 8
E = [(0,1),(0,3),(1,2),(2,3),(1,3),(0,2),(2,4),(3,4),(3,5),(2,6),(4,6),(4,5),(5,7),(6,7),
     (1,0),(3,0),(2,1),(3,2),(3,1),(2,0),(4,2),(4,3),(5,3),(6,2),(6,4),(5,4),(7,5),(7,6)]

# [verde, amarillo, rojo, azul]     
aik = [[1,2,0,0],
       [0,1,2,0],
       [2,0,1,2],
       [2,2,1,0],
       [2,0,1,1],
       [0,1,2,0],
       [1,0,0,2],
       [0,0,1,1]]

#
# declare variables
#

literals={}
for k in range(colores):
  for (i,j) in E:
    literals[i, j, k] = model.NewIntVar(0, 1, "x[%i,%i,%i]" % (i, j, k))

#
# constraints
#

# make sure that every arc is assigned exactly one color
[model.Add(sum([literals[i, j, k] for k in range(colores)]) == 1) for (i, j) in E]

# arcs (i, j) and ( j, i) must receive the same color (this is only necessary because the model is based on a directed graph).
[model.Add(literals[i, j, k]  == literals[j, i, k]) for (i, j) in E for k in range(colores)]

# check that the number of adjacent edges of each color is correct.
[model.Add(sum([literals[i, j, k] for (i, j) in E if i == l]) == aik[l][k]) for l in range(nodos) for k in range(colores)]

#
# search and result
#

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)
#print(status == cp_model.OPTIMAL)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
  E2 = [(0,1),(0,3),(1,2),(2,3),(1,3),(0,2),(2,4),(3,4),(3,5),(2,6),(4,6),(4,5),(5,7),(6,7)]

  for k in range(colores):
    for (i, j) in E2:
      if solver.Value(literals[i, j, k]) == 1:
        if k == 0:
          print(i, '-', j, 'verde')
        if k == 1:
          print(i, '-', j, 'amarillo')
        if k == 2:
          print(i, '-', j, 'rojo')
        if k == 3:
          print(i, '-', j, 'azul')