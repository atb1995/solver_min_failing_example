from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

nlayers = 10  # horizontal layers
ncols = 10  # number of columns
Lx = 1000.0
Lz = 1000.0
mesh_name = 'dry_compressible_mesh'
m = PeriodicIntervalMesh(ncols, Lx)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers, name=mesh_name)

V = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, z= SpatialCoordinate(mesh)

velocity = as_vector((0.5, 0.0))
u = Function(W).interpolate(velocity)

g = Constant(9.810616)
R_d =Constant(287.)

T = Constant(300.0)
zH = R_d * T / g
p = Constant(100000.0) * exp(-z / zH)

q = Function(V).interpolate(p / (R_d * T))
q_init = Function(V).assign(q)

qs = []

dt = 6.0
t_max = 50*dt
dtc = Constant(dt)
q_in = Constant(1.0)

dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*dq_trial*dx

dq = Function(V)

n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))
dS_ = (dS_v + dS_h) 
L1 = -inner(grad(phi), outer(q, u))*dx  + dot(jump(phi), (un('+')*q('+') - un('-')*q('-')))*dS_
R1 = phi*dq*dx-inner(grad(phi), outer(q, u))*dx  + dot(jump(phi), (un('+')*q('+') - un('-')*q('-')))*dS_

q1 = Function(V); q2 = Function(V); q3 = Function(V)
k1 = Function(V); k2 = Function(V); k3=Function(V)

L2 = replace(L1, {q: q1}); L3 = replace(L1, {q: q2}); L4 = replace(L1, {q: q3})

R2 = replace(R1, {q: q1}); R3 = replace(R1, {q: q2}); R4 = replace(R1, {q: q3})

params = {'ksp_type': 'cg', 'ksp_monitor':None, 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(a, L1, dq)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dq)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a, L3, dq)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)
prob4 = LinearVariationalProblem(a, L4, dq)
solv4 = LinearVariationalSolver(prob4, solver_parameters=params)


prob1 = NonlinearVariationalProblem(R1, dq)
solv1 = NonlinearVariationalSolver(prob1, solver_parameters=params)
prob2 = NonlinearVariationalProblem(R2, dq)
solv2 = NonlinearVariationalSolver(prob2, solver_parameters=params)
prob3 = NonlinearVariationalProblem(R3, dq)
solv3 = NonlinearVariationalSolver(prob3, solver_parameters=params)
prob4 = NonlinearVariationalProblem(R4, dq)
solv4 = NonlinearVariationalSolver(prob4, solver_parameters=params)

t = 0.0
step = 0
output_freq = 2
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

    if step % output_freq == 0:
        qs.append(q.copy(deepcopy=True))
        print("t=", t)
print('max',np.max(q.dat.data[:]))
L2_err = sqrt(assemble((q - q_init)*(q - q_init)*dx))
L2_init = sqrt(assemble(q_init*q_init*dx))
print(L2_err/L2_init)
