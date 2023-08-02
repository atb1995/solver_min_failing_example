from firedrake import *
import math
import matplotlib.pyplot as plt
import numpy as np

# Minimal failing example: Nonlinear solver fails when linear solver does not.
# The test case is on a periodic mesh and uses a dry bubble initialisation.
# The time discretisation uses forward euler, but solved using a Butcher Tableau, as follows:
# k1 = f(q^(n))
# q^(n+1) = q^(n) + dt*k1

# Set up periodic mesh
nlayers = 10  # horizontal layers
ncols = 10  # number of columns
Lx = 1000.0
Lz = 1000.0
mesh = PeriodicSquareMesh(ncols, nlayers, Lx)

# Set up compatable spaces
V = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, z = SpatialCoordinate(mesh)

# Constant velocity and dry bubble initial conditions
velocity = as_vector((0.5, 0.0))
u = Function(W).interpolate(velocity)

# Function which varies in z only to return zero increment
q = Function(V).interpolate(z)

# Only a small number of timesteps requuired for error
dt = 6.0
t_max = 2*dt

# Upwind DG advection
dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*dq_trial*dx

dq = Function(V)

n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

# Form to send to linear solver
L = -inner(grad(phi), outer(q, u))*dx  + dot(jump(phi), (un('+')*q('+') - un('-')*q('-')))*dS

# Form to send to nonlinear solver
R = phi*dq*dx-inner(grad(phi), outer(q, u))*dx  + dot(jump(phi), (un('+')*q('+') - un('-')*q('-')))*dS

# Set up solver parameters with ksp monitor
params = {'ksp_type': 'cg', 'ksp_monitor':None, 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}

# Set up linear solver
prob = LinearVariationalProblem(a, L, dq)
solv = LinearVariationalSolver(prob, solver_parameters=params)

# Set up nonlinear solver
nl_prob = NonlinearVariationalProblem(R, dq)
nl_solv = NonlinearVariationalSolver(nl_prob, solver_parameters=params)

# Run for linear solver
t = 0.0
step = 0
while t < t_max:
    solv.solve()
    q.assign(q + dt*dq)

    step += 1
    t += dt

print("Linear solver run complete!")

# Run for nonlinear solver
t = 0.0
step = 0
while t < t_max:
    nl_solv.solve()
    q.assign(q + dt*dq)

    step += 1
    t += dt

print("Nonlinear solver run complete!")