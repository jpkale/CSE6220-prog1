/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include <vector>

double calcL2Norm(const int n, const double* A, const double* b, const double* x, double* y);

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
/**
 * @brief   Performs the matrix vector product: y = A*x
 *
 * @param n     The size of the dimensions.
 * @param A     A n-by-n matrix represented in row-major order.
 * @param x     The input vector of length n.
 * @param y     The output vector of length n.
 */
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    for (int i=0; i<n; i++) {
		y[i] = 0.0;
		
		for (int j=0; j<n; j++) {
			y[i] += A[(i*n)+j] * x[j];
		}
	}
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
/**
 * @brief   Performs the matrix vector product: y = A*x
 *
 * @param n     The size of the first dimension.
 * @param m     The size of the second dimension.
 * @param A     A n-by-m matrix represented in row-major order.
 * @param x     The input vector of length m.
 * @param y     The output vector of length n.
 */
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
	for (int i=0; i<n; i++) {
		y[i] = 0.0;
		
		for (int j=0; j<m; j++) {
			y[i] += A[i*m +j] * x[j];
		}
	}
}

// implements the sequential jacobi method
/**
 * @brief   Performs Jacobi's method for solving A*x=b for x.
 *
 * @param n                 The size of the input.
 * @param A                 The input matrix `A` of size n-by-n.
 * @param b                 The input vector `b` of size n.
 * @param x                 The output vector `x` of size n.
 * @param max_iter          The maximum number of iterations to run.
 * @param l2_termination    The termination criteria for the L2-norm of
 *                          ||Ax - b||. Terminates as soon as the total L2-norm
 *                          is smaller or equal to this.
 */
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    std::vector<double> D(n);
	std::vector<double> R(A, A+n*n);
	
	for (int i=0; i<n; i++) {
		//D = diag(A)
		D[i] = A[n*i + i];
		// R = A-D
		R[n*i + i] = 0.0;
	}
	
	int iterCount = 0;
	double norm = 0.0;
	std::vector<double> temp(n);
	
	while (iterCount < max_iter) {
		//find l2-norm
		norm = calcL2Norm(n, A, b, x, &temp[0]);
		
		//compare norm against l2_termination
		if (norm <= l2_termination)
			break;
		
		// x = (b-Rx)/d
		matrix_vector_mult(n, &R[0], x, &temp[0]);
		for (int i=0; i<n; i++) {
			x[i] = (b[i] - temp[i]) / D[i];
		}
		iterCount++;
	}
}

// performs an L2-norm calculation ||Ax-b||
double calcL2Norm(const int n, const double* A, const double* b, const double* x, double* y) {
	double result = 0.0;
	matrix_vector_mult(n,A,x,y);
	
	for (int i=0; i<n; i++) {
		result = result + pow(y[i]-b[i], 2.0);
	}
	
	return sqrt(result);
}
