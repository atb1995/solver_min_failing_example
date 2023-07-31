from firedrake import *
import math
import matplotlib.pyplot as plt
import numpy as np

# Minimal failing example: Nonlinear solver fails when linear solver does not.
# The test case is on a periodic mesh and uses a dry bubble initialisation.
# The time discretisation uses RK4 solved as follows:
# k1 = f(q^(n))
# k2 = f(q^(n) + dt/2 *k1)
# k3 = f(q^(n) + dt/2 *k2)
# k4 = f(q^(n) + dt*k3)
# q^(n+1) = q^(n) + dt/6*(k4 + 2*k3 + 2*k2 + k1)

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

g = Constant(9.810616)
R_d =Constant(287.)

T = Constant(300.0)
zH = R_d * T / g
p = Constant(100000.0) * exp(-z / zH)

q = Function(V).interpolate(p / (R_d * T))
q_init = Function(V).assign(q)

# Only a small number of timesteps requuired for error
dt = 6.0
t_max = 2*dt
q_in = Constant(1.0)

# Upwind DG advection
dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*dq_trial*dx

dq = Function(V)

n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

# Fields at different stages
q1 = Function(V); q2 = Function(V); q3 = Function(V)
k1 = Function(V); k2 = Function(V); k3=Function(V)

# Form to send to linear solver
L1 = -inner(grad(phi), outer(q, u))*dx  + dot(jump(phi), (un('+')*q('+') - un('-')*q('-')))*dS
L2 = replace(L1, {q: q1}); L3 = replace(L1, {q: q2}); L4 = replace(L1, {q: q3})

# Form to send to nonlinear solver
R1 = phi*dq*dx-inner(grad(phi), outer(q, u))*dx  + dot(jump(phi), (un('+')*q('+') - un('-')*q('-')))*dS
R2 = replace(R1, {q: q1}); R3 = replace(R1, {q: q2}); R4 = replace(R1, {q: q3})

# Set up solver parameters with ksp monitor
params = {'ksp_type': 'cg', 'ksp_monitor':None, 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}

# Set up linear solver
prob1 = LinearVariationalProblem(a, L1, dq)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dq)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a, L3, dq)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)
prob4 = LinearVariationalProblem(a, L4, dq)
solv4 = LinearVariationalSolver(prob4, solver_parameters=params)

# Set up nonlinear solver
nl_prob1 = NonlinearVariationalProblem(R1, dq)
nl_solv1 = NonlinearVariationalSolver(nl_prob1, solver_parameters=params)
nl_prob2 = NonlinearVariationalProblem(R2, dq)
nl_solv2 = NonlinearVariationalSolver(nl_prob2, solver_parameters=params)
nl_prob3 = NonlinearVariationalProblem(R3, dq)
nl_solv3 = NonlinearVariationalSolver(nl_prob3, solver_parameters=params)
nl_prob4 = NonlinearVariationalProblem(R4, dq)
nl_solv4 = NonlinearVariationalSolver(nl_prob4, solver_parameters=params)

# Run for linear solver
t = 0.0
step = 0
while t < t_max:
    solv1.solve()
    k1.assign(dq)
    q1.assign(q + 0.5*dt*dq)

    solv2.solve()
    k2.assign(dq)
    q2.assign(q + 0.5*dt*dq)

    solv3.solve()
    k3.assign(dq)
    q3.assign(q + dt*dq)

    solv4.solve()
    q.assign(q + (1./6.)*dt*(dq+2.*k3 +2.*k2+k1))

    step += 1
    t += dt

print("Linear solver run complete!")

# Run for nonlinear solver
t = 0.0
step = 0
while t < t_max:
    nl_solv1.solve()
    k1.assign(dq)
    q1.assign(q + 0.5*dt*dq)

    nl_solv2.solve()
    k2.assign(dq)
    q2.assign(q + 0.5*dt*dq)

    nl_solv3.solve()
    k3.assign(dq)
    q3.assign(q + dt*dq)

    nl_solv4.solve()
    q.assign(q + (1./6.)*dt*(dq+2.*k3 +2.*k2+k1))

    step += 1
    t += dt

print("Nonlinear solver run complete!")