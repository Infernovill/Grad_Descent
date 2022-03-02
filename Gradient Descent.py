#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as np
Salary = np.read_csv(r"C:\Users\rafyq\OneDrive\Desktop\Sal_vs_Exp.csv")
Salary


# In[13]:


# objective function
def objective(x = 'Years of Experience', y = 'Salary'):
    return x + y


# In[14]:


# 3d plot of the test function
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot

# define range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.gca(projection='3d')
axis.plot_surface(x, y, results, cmap='jet')
# show the plot
pyplot.show()


# In[15]:


# contour plot of the test function
from numpy import asarray
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot


# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# show the plot
pyplot.show()


# In[21]:


# gradient descent optimization with nesterov momentum for a two-dimensional test function
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed



# derivative of objective function
def derivative(x, y):
    return asarray([x, y])

# gradient descent algorithm with nesterov momentum
def nesterov(objective, derivative, bounds, n_iter, step_size, momentum):
# generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
# list of changes made to each variable
    change = [0.0 for _ in range(bounds.shape[0])]
# run the gradient descent
    for it in range(n_iter):
# calculate the projected solution
        projected = [solution[i] + momentum * change[i] for i in range(solution.shape[0])]
# calculate the gradient for the projection
        gradient = derivative(projected[0], projected[1])
# build a solution one variable at a time
        new_solution = list()
        for i in range(solution.shape[0]):
# calculate the change
            change[i] = (momentum * change[i]) - step_size * gradient[i]
# calculate the new position in this variable
            value = solution[i] + change[i]
# store this variable
            new_solution.append(value)
# evaluate candidate point
        solution = asarray(new_solution)
        solution_eval = objective(solution[0], solution[1])
# report progress
        print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
    return [solution, solution_eval]

# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 30
# define the step size
step_size = 0.1
# define momentum
momentum = 0.3
# perform the gradient descent search with nesterov momentum
best, score = nesterov(objective, derivative, bounds, n_iter, step_size, momentum)
print('Done!')
print('f(%s) = %f' % (best, score))


# In[22]:


# gradient descent algorithm with nesterov momentum
def nesterov(objective, derivative, bounds, n_iter, step_size, momentum):
# track all solutions
    solutions = list()
# generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
# list of changes made to each variable
    change = [0.0 for _ in range(bounds.shape[0])]
# run the gradient descent
    for it in range(n_iter):
# calculate the projected solution
        projected = [solution[i] + momentum * change[i] for i in range(solution.shape[0])]
# calculate the gradient for the projection
        gradient = derivative(projected[0], projected[1])
# build a solution one variable at a time
        new_solution = list()
        for i in range(solution.shape[0]):
# calculate the change
            change[i] = (momentum * change[i]) - step_size * gradient[i]
# calculate the new position in this variable
            value = solution[i] + change[i]
# store this variable
            new_solution.append(value)
# store the new solution
        solution = asarray(new_solution)
        solutions.append(solution)
# evaluate candidate point
        solution_eval = objective(solution[0], solution[1])
# report progress
        print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
    return solutions


# In[18]:


# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# define the step size
step_size = 0.01
# define momentum
momentum = 0.8
# perform the gradient descent search with nesterov momentum
solutions = nesterov(objective, derivative, bounds, n_iter, step_size, momentum)


# In[19]:


# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')


# In[24]:



# example of plotting the nesterov momentum search on a contour plot of the test function
from math import sqrt
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
 
# objective function
def objective(x, y):
    return x + y
 
# derivative of objective function
def derivative(x, y):
    return asarray([x , y ])
 
# gradient descent algorithm with nesterov momentum
def nesterov(objective, derivative, bounds, n_iter, step_size, momentum):
# track all solutions
    solutions = list()
# generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
# list of changes made to each variable
    change = [0.0 for _ in range(bounds.shape[0])]
# run the gradient descent
    for it in range(n_iter):
# calculate the projected solution
        projected = [solution[i] + momentum * change[i] for i in range(solution.shape[0])]
# calculate the gradient for the projection
        gradient = derivative(projected[0], projected[1])
# build a solution one variable at a time
        new_solution = list()
        for i in range(solution.shape[0]):
# calculate the change
            change[i] = (momentum * change[i]) - step_size * gradient[i]
# calculate the new position in this variable
            value = solution[i] + change[i]
# store this variable
            new_solution.append(value)
# store the new solution
        solution = asarray(new_solution)
        solutions.append(solution)
# evaluate candidate point
        solution_eval = objective(solution[0], solution[1])
# report progress
        print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
    return solutions
 
# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# define the step size
step_size = 0.01
# define momentum
momentum = 0.8
# perform the gradient descent search with nesterov momentum
solutions = nesterov(objective, derivative, bounds, n_iter, step_size, momentum)
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# plot the sample as black circles
solutions = asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# show the plot
pyplot.show()


# In[ ]:




