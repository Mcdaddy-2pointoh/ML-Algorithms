#version: 1.0.0
#creator: BaceFook
#Description: This py file contains a simplified regression algorithm with cost function, hypothesis function and a gradient descent approach

"""
*********************     Basic definitions     *****************************

m => number of instances/data points
n => number of features (starts from n)
h => estimated value using theeta(j) and x
t(j) => jth parameter of regression (also commonly refered to as theeta)
y => Actual value of the given data corresponding to x
a => learning rate
X => Matrix of dimension m x n
Y => Vector of order n i.e matrix of order n x 1

**********************    Simplified formulae    ****************************

1. LINEAR HYPOTHESIS FUNCTION:
h = t0(x0) + t1(x1) + t2(x2) ..... basis the number of features (n)

2. COST FUNCTION:
J = 1/2m [Summation from i = 1 : 1 : m ((h - y)^2)]

3. GRADIENT DESCENT: 
t(j) = t(j) - a[1/m(Summation from i = 1 : 1 : m ((h - y(i))(x(i)))]

**********************    Developers reccomendation    *********************

1. Do not alter variable values
2. Insert all values in gradient descent and other functions as vectors (n x 1 matrice).
3. Increase learning rates if gradient descent is slow. Reccomended strp size is 3x.
"""
import numpy as np


# This is the gradient descent function in order to minimize cost
def gradient_descent(ji , H, T, X, Y, a):
    try:
        [m, n] = X.shape
        Jd = []
        for d in range(0, n, 1):
            Xn = np.array([X[0:,d]])
            Hd = (H-Y)
            Jdn = 1/(m+1)*(np.matmul(Xn, Hd))
            Jd.append([Jdn[0,0]])
        Jd = np.array(Jd)
        T = T - a*Jd
        H = np.matmul(X, T)
        J = (1/(2*(m+1)))*((H-Y)**2)
        j = np.matmul(O, J)
        if j[0,0] < ji[0,0] :
            a = a*2.25
            T = gradient_descent(j, H, T, X, Y, a)
            return T

        elif (j[0,0] == 0) or (j[0,0] == ji[0,0]):
            return T

        elif j[0, 0 ] > ji[0, 0]:
            a = a/3
            T = gradient_descent(j, H, T, X, Y, a)
          
            return T
    except RecursionError:
        return T

      
# This is the cost function, this needs to be minimized and must tend to 0 in order to achieve profeciency 
def cost_function(H, T, X, Y, a):
    [m,_] = X.shape
    global O
    O = np.ones((1, m))
    J = (1/(2*(m+1)))*((H-Y)**2)
    j = np.matmul(O, J)
    T = gradient_descent(j, H, T, X, Y, a)
    H = np.matmul(X, T)
    print (T)
    print (H)
    print (Y)
    return [H,T]
  

# This is the linear hy pothesis function commonly called a regression line to fit to multiple data points
def linear_hypothesis(X, Y, a=0.0003):
    n = X.shape[1]
    if X.shape[0] != Y.shape[0]:
        print("ERROR: Incompatible X and Y matrix ensure the number of rows in each are the same")

    elif Y.shape[1] != 1:
        print("ERROR: The provided matrix Y is not a vector, please change the matrix to a m x 1 format")

    else:
        print("You have set {} features in your regression algorithm" .format(n))
        T = []
        global Z
        Z = []
        for _ in range(0, n, 1):
            T.append(0)
            Z.append(0)
        Z = np.array([Z]).transpose()
        T = np.array([T]).transpose()
        H = np.matmul(X, T)
        [H , T] = cost_function(H, T, X, Y, a)
        E = Y - H
        #h = theeta transpose * X
        return [H, T, E]

#Dummy Data, alter your data in A and B, alpha is user dependent with a default of 0.0003
A = np.array([[307.0, 130.0, 3504, 12.0], [350.0, 165.0, 3693, 11.5 ], [318.0, 150.0, 3436, 11.0], [304.0, 150.0, 3433, 12.0], [302.0, 140.0, 3449, 10.5], [429.0, 198.0, 4341, 10.0], [454.0, 220.0, 4354, 9.0], [440.0, 215.0, 4312, 8.5], [455.0, 225.0, 4425, 10.0], [390.0, 190.0, 3850, 8.5]])
B = np.array([[18],[15],[18],[16],[17],[15],[14],[14],[14],[15]])
[Hypothesis, Theeta, ErrorDifference] = linear_hypothesis(A, B, 0.0006235)


