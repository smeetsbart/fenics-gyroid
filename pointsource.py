from fenics import *
import os, pickle, time, math
import numpy as np

# Create mesh and define function space
nx = ny = nz = 99
#mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
pi=math.pi

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

def save_params( params ):
   with open("params.pickle",'wb') as fob:
      pickle.dump(params, fob)

pi_4 = f"{0.25*pi/4}"

xarg = f"w*x[0]+{pi_4}"
yarg = f"w*x[1]+{pi_4}"
zarg = f"w*x[2]+{pi_4}"

t2 = f"( abs(sin( {xarg} ) * cos( {yarg}  ) + sin( {yarg} )*cos( {zarg} ) + sin( {zarg} )*cos( {xarg} )) / 1.5 )"
#Here we make a single line expression of the gyroid equation:
exprstr = f"({t2} <= h ) ? ( v1 ) : ( v1 + ( v2-v1 ) * sqrt( ({t2}-h)/(1-h) ) )"
maskstr = f"({t2} <= h ) ? 0 : 1"

print(f"Gyroid expression: {exprstr}")

# Create VTK file for saving solution
cfile = File('diffusion_gaussian/concentration.pvd')#One for every output step
rfile = File("diffusion_gaussian/absorption_rate.pvd")#We will make only one
dfile = File("diffusion_gaussian/diffusivity.pvd")#We will make only one
afile = File("diffusion_gaussian/absorbed.pvd")#One for every output step
sfile = File("diffusion_gaussian/source.pvd")

#Diffusion length scale in eps
Leps = pi/20
#Base diffusion in water, we set to one to set the timescale
D0 = 1.
#Relative diffusivity of EPS
Deps = 0.01
r0 = 0.#No absorption in water
void_fraction = 0.45
reps = Deps / Leps**2
Lss = [0.2*pi, pi]

params = { "L_eps" : Leps
         , "D0" : D0
         , "D_eps" : Deps
         , "r0" : r0
         , "void_fraction" : void_fraction
         , "r_eps" : reps
         , "Ls" : Lss }

save_params( params )

for Ls in Lss:

   print(f"Solving for Ls = {Ls:1.2f}")
   w = 2*pi / ( 2*Ls)

   mesh = BoxMesh( Point(-pi,-pi,-pi),Point(pi,pi,pi) , nx,ny,nz )

   V = FunctionSpace(mesh, 'P', 1)

   bc = DirichletBC(V, Constant(0), boundary)

   a=25
   total_u = pi**(3/2) / ( a*a**0.5 )#-oo to +oo integral of initial expression
   #print(f"total u: {total_u}")
   # Define initial value
   f = Expression(f'(1/{total_u})*exp(-a*pow(x[0], 2) - a*pow(x[1], 2) - a*pow(x[2],2))',
                    degree=2, a=a)

   # Define variational problem
   u = TrialFunction(V)
   v = TestFunction(V)

   diff = Expression( exprstr, v1=D0,v2=Deps, w=w, h=void_fraction, degree=2 )
   r    = Expression( exprstr, v1=r0,v2=reps, w=w, h=void_fraction, degree=2 )
   mask = Expression( maskstr, w=w, h=void_fraction, degree=0)

   F = dot(diff*grad(u), grad(v))*dx + r * u*v*dx - f*v*dx
   a, L = lhs(F), rhs(F)

   rfile << interpolate(r, V)
   dfile << interpolate(diff, V)
   sfile << interpolate(f, V)

   u = Function(V, name='u')
   solver_parameters = {'linear_solver': 'gmres',
                        'preconditioner': 'ilu'}
   solve(a==L,u,bc, solver_parameters=solver_parameters )
   ac_n = Function(V,name='a')
   ac_n.assign(project(r*u, V, solver_type='gmres'))

   cfile << u
   afile << ac_n
