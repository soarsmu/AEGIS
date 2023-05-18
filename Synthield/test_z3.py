from z3 import *

# Create Z3 variables for the coefficients
a = Real('a')
b = Real('b')

# Create Z3 solver
solver = Solver()

# Define the linear function
x = Real('x')
f = a * x + b

# Specify constraints or properties
solver.add(f == 2 * x + 3)

# Solve the constraints
if solver.check() == sat:
    # Get the model
    model = solver.model()
    
    # Get the values of the coefficients
    coefficient_a = model.eval(a)
    coefficient_b = model.eval(b)
    
    print("Coefficient a:", coefficient_a)
    print("Coefficient b:", coefficient_b)
else:
    print("Constraints are unsatisfiable.")
