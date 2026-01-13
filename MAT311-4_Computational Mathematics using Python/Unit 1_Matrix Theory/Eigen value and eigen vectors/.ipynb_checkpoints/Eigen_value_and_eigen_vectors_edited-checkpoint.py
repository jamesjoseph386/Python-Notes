
# coding: utf-8

# # Eigenvalues and Eigenvectors

# Eigen vector of a matrix A is a vector represented by a matrix X such that when X is multiplied with matrix A, then the direction of the resultant matrix remains same as vector X.
# 
# Mathematically, above statement can be represented as:
# 
#                                     AX = λX
# 
# where A is any arbitrary matrix, λ are eigen values and X is an eigen vector corresponding to each eigen value.
# 
# (i)  The eigen values and corresponding eigen vectors are given by the characteristic equation ,
# 
#                                     |A – λI| = 0
#                           
# (ii) To find the eigen vectors, we use the equation (A – λI) X = 0  and solve it by Gaussian elimination, that is, convert the        augmented matrix (A – λI) = 0 to row echelon form and solve the linear system of equations thus obtained.  
# 
# 
# SYNTAX: np.linalg.eigvals(A) (Returns eigen values)
# 
# SYNTAX: np.linalg.eig(A)     (Returns eigen vectors)

# # To obtain eigenvalues and eigenvectors using python.

# 1. Write a python program to find the eigen values and eigen vectors for the given matrix.
# 
#                      [4,3,2],[1,4,1],[3,10,4]

# In[1]:


import numpy as np
I=np.array([[4,3,2],[1,4,1],[3,10,4]])
print("Given matrix")
print(I)
print()
x=np.linalg.eigvals(I)
y=np.linalg.eig(I)
print("Eigen values:")
print(x)
print()
print("Eigen vectors:")
print(y)


# 2.  Write a python program to find the eigen values and eigen vectors for the given matrix.
# 
#                                  [1,-3,3],[3,-5,3],[6,-6,4]

# In[2]:


import numpy as np
I=np.array([[1,-3,3],[3,-5,3],[6,-6,4]])
print("Given Matrix")
print(I)
print()
x=np.linalg.eigvals(I)
y=np.linalg.eig(I)
print("Eigen values:")
print(x)
print()
print("Eigen vectors:")
print(y)


# # Properties of eigenvalues and eigenvectors

# 
# 1. For a nxn matrix, the number of eigen values is n.
# 2. The sum of eigen values is equal to the sum of the diagonal elements of matrix.
# 3. The product of eigenvalues is equal to the determinant of the matrix.
# 4. The eigen value for an identity matrix is 1.
# 5. The eigenvalue of a triangular matrix is same as the diagonal elements of a matrix.
# 6. For a skew symmetric matrix, the eigenvalues are imaginary.
# 7. For orthogonal matrix the length of eigenvalues equal to 1.
# 8. For indempotent matrix the eigenvalues are 0 and 1(Aˆ2=identity matrix).

# In[3]:


import math
from math import *
import numpy as np
from numpy import *


# #### Property 2

# In[4]:


A=np.array([[1,2,3],[2,3,5],[3,1,1]])
print("Given matrix")
print(A)
print()
X=float(A[0][0]+A[1][1]+A[2][2])
print("The sum of the diagonal elements of matrix ")
print(X)
print()
Y=sum(np.linalg.eigvals(A))
Z=round(Y)
print("The sum of eigen values")
print(Z)
print()
print("The sum of eigen values is equal to the sum of the diagonal elements of matrix")


# #### Property 3

# In[5]:


B=np.array([[1,2,3],[1,3,5],[4,1,2]])
print(" Given matrix")
print(B)
print()
M=np.round(np.linalg.det(B))
print("The determinant of the matrix")
print(M)
print()
Q=np.prod(np.linalg.eigvals(B))
P=round(Q)          
print("The product of eigenvalues")
print(P)
print()
print("The product of eigenvalues is equal to the determinant of the matrix.")


# #### Property 4

# In[6]:


I=np.array([[1,0,0],[0,1,0],[0,0,1]])
print("#### Input matrix")
print(I)
print()
print("The eigen value for an identity matrix is 1")
np.linalg.eigvals(I)


# Property 5

# In[7]:


T=np.array([[4,0,0],[2,3,0],[1,2,3]])
print("Given matrix")
print(T)
print()
print("The eigenvalue of a triangular matrix is same as the diagonal elements of the matrix.")
np.linalg.eigvals(T)


# Property 6

# In[8]:


B=np.array([[1,2,3],[1,3,5],[4,1,2]])
print(" Given matrix")
print(B)
print()
E=(B-B.transpose())/2
print(" Skew symmetric matrix")
print(E)
print()
print("For a skew symmetric matrix, the eigenvalues are imaginary.")
np.linalg.eigvals(E)


# Property 7

# In[9]:


F=np.array([[1,0],[0,-1]])
print("Orthogonal matrix")
print(F)
print()
print("For orthogonal the magnitude of each  eigenvalue is 1.(Orthogonal matrix---> A.(A)ˆt=I).")
np.linalg.eigvals(F)


# # Diagonalization of square matrix

# Given the matrix A, let the matrix of eigen vectors of A be P and its inverse OF P be I.
# Then the diagonalised matrix is given by the equation,
#                                     
#                                        IAP = D                                  

# In[2]:


import numpy as np
from math import *
A= np.mat([[2,-2,3],[1,1,1],[1,3,-1]])
X,P=np.linalg.eig(A)
I=np.linalg.inv(P)
Z=np.around(I*A*P)
for i in range(len(Z)):
    for j in range(len(Z)):
        if Z[i,j]==-0:
            Z[i,j]=0

print("The final diagonalized matrix is")
print(Z)
print()

print("Eigen vectors")
print(P)
print()

print("Eigen values")
print(X)
print()


# # Cayley-Hamilton Theorem

# Every non singular square matrix satisfy its characteristic equation.

# In[13]:


A=np.mat([[2, 3],[4, 5]])
from math import *
X=np.poly(A)
print("The co-efficients of the characteristic equation are",X)

trace_A = np.trace(A)
print(trace_A)
det_A = np.linalg.det(A)
print(det_A) 
I = np.eye(len(A))
print(I)
P=A*A - trace_A * A + det_A * I
print(P)

from sympy import *
from math import *
from numpy import *
print("Enter elements of the matrix: ")
A=mat(input())
s=0
print()
print("The matrix is: ")
print(A)
print()
I=eye(len(A),len(A))
print("The identity matrix is: ")
print(I)
print()
ce=poly(A)
ce=ce.round()
print("The coefficients of the characteristic equation=",ce)
for i in range (len(ce)):
    eq=ce[i]*I*(A**(len(ce)-i))
    s=s+eq
print()
s

