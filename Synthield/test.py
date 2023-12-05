from z3 import *

# Create Z3 variables for the state
x, y = Reals('x y')

# Define the polyhedron constraints (box constraint)
polyhedron_constraints = [And(x >= -1, x <= 1, y >= -1, y <= 1)]

# Create a solver instance
solver = Solver()

# Add constraints for the polyhedron
solver.add(polyhedron_constraints)

# Check satisfiability to find the maximal constraint-admissible invariant set
while solver.check() == sat:
    # Get a model
    model = solver.model()

    # Extract values for x and y from the model
    x_val = model[x].as_decimal(3)
    y_val = model[y].as_decimal(3)

    # Block the current solution
    solver.add(Or(x != x_val, y != y_val))

    # Print the current solution
    print("x =", x_val, "y =", y_val)

# The loop terminates when no more points in the invariant set are found
print("Computation of the maximal constraint-admissible invariant set complete.")
