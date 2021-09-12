import numpy as np
from numpy import array
# Vectors
# Arrays describing a space
a = array([1,2,3])
print("\nVector a : \n", a)
b = array([4,5,6])
print("\nVector b : \n", b)
# vector addition
vAddition = a+b
print("\nVector a + b : \n", vAddition)
# vector subtraction
vSubtraction = a-b
print("\nVector a - b: \n", vSubtraction)
# vector multiplication
vMultiplicaton = a*b
print("\nVector a * b: \n", vMultiplicaton)
# vector division
vDivision = a/b
print("\nVector a / b: \n", vDivision)
# vector dot product
vDotProd = a.dot(b)
print("\nVector a dotprod b: \n", vDotProd)


# Matrices
# Basically Multi-dim vectors
A = array([[1,2,3],[4,5,6],[7,8,9]])
print("\nMatrix A: \n", A)
B = array([[7,8,9],[4,5,6],[1,2,3]])
print("\nMatrix B: \n", B)
# Matrix Addition
mAddition = A + B
print("\nMatrix A + B: \n", mAddition)
# Matrix Subtraction
mSubtraction = A-B
print("\nMatrix A - B: \n", mSubtraction)
# Matrix multiplication
mMultiplicaton = np.multiply(A,B)
print("\nMatrix A * B: \n", mMultiplicaton)
# Matrix Division
mDivision = np.divide(A, B)
print("\nMatrix A / B: \n", mDivision)

# Matrix DotProd
mDotProd = A.dot(B)
print("\nMatrix A dotprod B: \n", mDotProd)

############
# Matrix types and operations
# Transpose C = A^T
mTranspose = A.T
print("\nMatrix A transposed: \n", mTranspose)

# Invert C = A^-1
mInvert = np.linalg.inv(A)
print("\nMatrix A inverted: \n", mInvert)

# Trace of A
mTrace = np.trace(A)
print("\nMatrix A trace: \n", mTrace)

# Determinant of A
mDeterminant = np.linalg.det(A)
print("\nMatrix A determinant: \n", mDeterminant)

# Rank of A
mRank = np.linalg.matrix_rank(A)
print("\nMatrix A rank: \n", mRank)






