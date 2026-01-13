#!/usr/bin/env python
# coding: utf-8

# # Solving System of Equations

# A linear equation in variables $x_1, x_2, \ldots, x_n$ is an equation of the form $a_1x_1 + a_2x_2 + \ldots + a_nx_n = b$, where $a_1, a_2, \ldots, a_n$ and $b$ are constants. The constant $a_i$ is called the coefficient of $x_i$.

# A system of linear equations is a finite collection of linear equations in same variables. <br> The following is a system of $m$ linear equations in $n$ variables $x_1, x_2, \ldots, x_n$.<br>
# $a_{11}x_1 + a_{12}x_2 + \ldots + a_{1n}x_n = b_1$ <br>
# $a_{21}x_1 + a_{22}x_2 + \ldots + a_{2n}x_n = b_2$ <br>
# $\ldots$ <br>
# $a_{m1}x_1 + a_{m2}x_2 + \ldots + a_{mn}x_n = b_m$

# A solution of a linear system is an assignment of values to the variables $x_1, x_2, \ldots, x_n$ such that each of the equations is satisfied. The set of all solutions of a linear system is called the solution set of the system.

# In matrix notation a linear system is $AX=B$, where <br>
# $A=\begin{bmatrix} 
# a_{11} & a_{12} & \ldots & a_{1n} \\
# a_{21} & a_{22} & \ldots & a_{2n} \\
#  & \ldots &  \\
# a_{m1} & a_{m2} & \ldots & a_{mn} \\
# \end{bmatrix}$ is the coefficient matrix, <br>
# $X=\begin{bmatrix} 
# x_1 \\
# x_2 \\
# \vdots  \\
# x_n \\
# \end{bmatrix}$ and 
# $B=\begin{bmatrix} 
# b_1 \\
# b_2 \\
# \vdots  \\
# b_m \\
# \end{bmatrix}$.

# Python provides the function numpy.linalg.solve() to solve the system of linear equations. <br>
# $\textbf{Syntax}:$ <br>
# numpy.linalg.solve(A,B)

# Q. Find all solutions for the linear system <br>
# $x_1 + 2x_2 -x_3 = 1$ <br>
# $2x_1 + x_2 + 4x_3 = 2$ <br>
# $3x_1 + 3x_2 + 4x_3 = 1$

# In[15]:


import numpy as np


# In[16]:


A=np.array([[1,2,-1],[2,1,4],[3,3,4]])
B=np.array([1,2,1])
np.linalg.solve(A,B)


# Here we need to observe that the matrix $B$ is defined as a $ 1 \times 3 $ array, whereas the same function will not work if $B$ is defined as a $ 1 \times 3 $ matrix.

# In[17]:


A=np.matrix([[1,2,-1],[2,1,4],[3,3,4]])
B=np.matrix([1,2,1])
np.linalg.solve(A,B)


# This is because $B$ is a $m \times 1$ matrix in the matrix equivalent of the linear system of equation.

# In[32]:


A=np.matrix([[1,2,-1],[2,1,4],[3,3,4]])
B=np.matrix([[1],[2],[1]])
np.linalg.solve(A,B)


# In[33]:


A=np.matrix([[1,-1,1,-1],[1,-1,1,1],[4,-4,4,0],[-2,2,-2,1]])
B=np.matrix([[2],[0],[4],[-3]])
np.linalg.solve(A,B)


# In[6]:


A=np.matrix([[2,3,1],[1,-1,-2]])
B=np.matrix([[0],[0]])
np.linalg.solve(A,B)


# Note: The function np.linalg.solve() works only if $A$ is a non-singular matrix.

# ### System of Homogenous Linear Equations

# The linear system of equations of the form $AX=0$ is called system of homogenous linear equations. <br>
# The $n$-tuple $(0,0, \ldots, 0)$ is a trivial solution of the system. <br>
# The homogeneous system of $m$ equations $AX = 0$ in $n$ unknowns has a non trivial solution if and only if the rank of the matrix $A$ is less than $n$. Further if $\rho(A)=r < n$, then the system possesses $(n - r)$ linearly independent solutions.

# Q. Check whether the following system of homogenous linear equation has non-trivial solution. <br>
# $x_1 + 2x_2 -x_3 = 0$ <br>
# $2x_1 + x_2 + 4x_3 = 0$ <br>
# $3x_1 + 3x_2 + 4x_3 = 0$

# In[30]:


A=np.matrix([[1,2,-1],[2,1,4],[3,3,4]])
B=np.matrix([[0],[0],[0]])
r=np.linalg.matrix_rank(A)
n=A.shape[1]
if (r==n):
    print("System has trivial solution")
else:
    print("System has", n-r, "non-trivial solution(s)")


# In[31]:


np.linalg.solve(A,B)


# Q. Check whether the following system of homogenous linear equation has non-trivial solution. <br>
# $x_1 + 2x_2 -x_3 = 0$ <br>
# $2x_1 + x_2 + 4x_3 = 0$ <br>
# $x_1 - x_2 + 5x_3 = 0$

# In[29]:


A=np.matrix([[1,2,-1],[2,1,4],[1,-1,5]])
B=np.matrix([[0],[0],[0]])
r=np.linalg.matrix_rank(A)
n=A.shape[1]
if (r==n):
    print("System has trivial solution")
else:
    print("System has", n-r, "non-trivial solution(s)")


# ### System of Non-homogenous Linear Equations

# The linear system of equations of the form $AX=B$ is called system of non-homogenous linear equations if not all elements in $B$ are zeros. <br>
# The non homogeneous system of $m$ equations $AX=B$ in $n$ unknowns is consistent (has a solution) if and only if the $\rho(A) = \rho([A|B])$. <br>
# If $\rho(A) = \rho([A|B])$, and
# 1. $\rho(A) = n$, then system has unique solution
# 2. $\rho(A) < n$, then system has infintely many solutions. <br> 
# 
# If $\rho(A) \neq \rho([A|B])$, then the system is inconsistent.

# Q. Examine the consistency of the following system of equations and solve if consistent  <br>
# $x_1 + 2x_2 -x_3 = 1$ <br>
# $2x_1 + x_2 + 4x_3 = 2$ <br>
# $3x_1 + 3x_2 + 4x_3 = 1$

# In[28]:



A=np.matrix([[1,2,-1],[2,1,4],[3,3,4]])
B=np.matrix([[1],[2],[1]])
AB=np.concatenate((A,B), axis=1)
rA=np.linalg.matrix_rank(A)
rAB=np.linalg.matrix_rank(AB)
n=A.shape[1]
if (rA==rAB):
    if (rA==n):
        print("The system has unique solution")
        print(np.linalg.solve(A,B))
    else:
        print("The system has infinitely many solutions")
else:
    print("The system of equations is inconsistent")


# Q. Examine the consistency of the following system of equations and solve if consistent  <br>
# $x_1 + 2x_2 -x_3 = 1$ <br>
# $2x_1 + x_2 + 5x_3 = 2$ <br>
# $3x_1 + 3x_2 + 4x_3 = 1$

# In[27]:


A=np.matrix([[1,2,-1],[2,1,5],[3,3,4]])
B=np.matrix([[1],[2],[1]])
AB=np.concatenate((A,B), axis=1)
rA=np.linalg.matrix_rank(A)
rAB=np.linalg.matrix_rank(AB)
n=A.shape[1]
if (rA==rAB):
    if (rA==n):
        print("The system has unique solution")
        print(np.linalg.solve(A,B))
    else:
        print("The system has infinitely many solutions")
else:
    print("The system of equations is inconsistent")


# Q. Examine the consistency of the following system of equations and solve if consistent  <br>
# $x_1 - x_2 + x_3 - x_4 = 2$  <br>
# $x_1 - x_2 + x_3 + x_4 = 0$  <br>
# $4x_1 - 4x_2 + 4x_3 = 4$  <br>
# $-2x_1 + 2x_2 - 2x_3 + x_4 = -3$

# In[26]:


A=np.matrix([[1,-1,1,-1],[1,-1,1,1],[4,-4,4,0],[-2,2,-2,1]])
B=np.matrix([[2],[0],[4],[-3]])
AB=np.concatenate((A,B), axis=1)
rA=np.linalg.matrix_rank(A)
rAB=np.linalg.matrix_rank(AB)
n=A.shape[1]
if (rA==rAB):
    if (rA==n):
        print("The system has unique solution")
        print(np.linalg.solve(A,B))
    else:
        print("The system has infinitely many solutions")
else:
    print("The system of equations is inconsistent")


# ## Gauss Jordan Method

# Q. Write a program to find the solution of a sysltem of linear equations using Gauss Jordan method.

# In[25]:


m = int(input("Enter the number of equations:")) 
n = int(input("Enter the number of variables:")) 
print("Enter the entries of the coefficient matrix in a single line (separated by space): ") 
elements = list(map(int, input().split())) 
A=np.matrix(elements).reshape(m, n) 
print("Enter the entries of the matrix B in a single line (separated by space): ") 
Belements = list(map(int, input().split())) 
B=np.matrix(Belements).reshape(m, 1)
AB=np.concatenate((A,B), axis=1)
rA=np.linalg.matrix_rank(A)
rAB=np.linalg.matrix_rank(AB)
n=A.shape[0]
if (rA==rAB):
    if (rA==n):
        print("The system has unique solution")
        for i in range (n):
            if (AB[i,i]==0):
                k=i+1
                while (AB[k,i]==0):
                    k=k+1
                AB[[i,k]]=AB[[k,i]]
            AB[i]=AB[i]/AB[i,i]
            for j in range (n):
                if (j!=i):
                    AB[j]= AB[j]-AB[j,i]*AB[i]
        Solution=AB[:n,n]
        print("Solution:")
        print(Solution)
    else:
        print("The system has infinitely many solutions")
else:
    print("The system of equations is inconsistent")


# ## Cramer's Rule

# Q. Write a prgram to solve a system of linear equations using Cramer's rule.

# In[22]:


import copy as cp


# In[23]:


A=np.matrix([[1,2,-1],[2,1,4],[3,3,4]])
B=np.matrix([[1],[2],[1]])
Ax = cp.deepcopy(A)
Ay = cp.deepcopy(A)
Az = cp.deepcopy(A)
Ax[:, 0]=B
Ay[:, 1]=B
Az[:, 2]=B
x=np.linalg.det(Ax)/np.linalg.det(A)
y=np.linalg.det(Ay)/np.linalg.det(A)
z=np.linalg.det(Az)/np.linalg.det(A)
print(" x={:.2f} \n y={:.2f} \n z={:.2f}".format(x,y,z))


# ## Gauss Elimination Method

# Q. Write a program to find the solution of a sysltem of linear equations using Gauss elimination method.

# In[21]:


A=np.matrix([[1,2,-1],[2,1,4],[3,3,4]])
B=np.matrix([[1],[2],[1]])
AB=np.concatenate((A,B), axis=1)
m=A.shape[0]
n=A.shape[1]
for i in range (m):
    if (AB[i,i]==0):
        k=i+1
        while (AB[k,i]==0):
            k=k+1
        AB[[i,k]]=AB[[k,i]]
    AB[i]=AB[i]/AB[i,i]
    for j in range (i+1,m):
        AB[j]= AB[j]-AB[j,i]*AB[i]
z = AB[2,3]/AB[2,2]
y = (AB[1,3]-z*AB[1,2])/AB[1,1]
x = (AB[0,3]-z*AB[0,2]-y*AB[0,1])/AB[0,0]
print("Solution:")
print(" x={:.2f} \n y={:.2f} \n z={:.2f}".format(x,y,z))


# In[ ]:




