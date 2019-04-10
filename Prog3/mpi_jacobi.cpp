/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */

 /*
 * @param n             The size of the input vector.
 * @param input_vector  The input vector of length `n`, only on processor (0,0).
 * @param local_vector  The local output vector of size floor(n/q) or
 *                      ceil(n/q), where `q` is the number of processors in the
 *                      first dimension of the 2d grid communicator. This
 *                      has to be allocated on the processors (i,0) according
 *                      to their block distirbuted size.
 * @param comm          A 2d cartesian grid communicator.
 */
void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    // TODO
}

/*
 * @param n             The total size of the output vector.
 * @param local_vector  The local input vector of size floor(n/q) or
 *                      ceil(n/q), where `q` is the number of processors in the
 *                      first dimension of the 2d grid communicator.
 * @param output_vector The output vector of length `n`, only on processor (0,0).
 * @param comm          A 2d cartesian grid communicator.
 */
// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    // TODO
}

/*
 * @param n             The size of the input dimensions.
 * @param input_matrix  The input matrix of size n-by-n, stored on processor
 *                      (0,0). This is an invalid pointer on all other processes.
 * @param local_matrix  The local output matrix of size n1-by-n2, where both n1
 *                      and n2 are given by the block distribution among rows
 *                      and columns. The output has to be allocated in this
 *                      process.
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 */
void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // TODO
}

/*
 * @param n             The total size of the input vector.
 * @param col_vector    The local vector as distributed among the first column
 *                      of processors. Has size ceil(n/q) or floor(n/q) on
 *                      processors (i,0). This is an invalid pointer (DO NOT
 *                      ACCESS) on processors (i,j) with j != 0.
 * @param row_vector    (Output) The local vector, block distributed among the
 *                      rows (see above example).
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 */
void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // TODO
}

/*
 * @param n             The size of the input dimensions.
 * @param local_A       The distributed matrix A.
 * @param local_x       The distirbuted input vector x, distirbuted among the
 *                      first column of the processor grid (i,0).
 * @param local_y       The distributed output vector y, distributed among the
 *                      first column of the processor grid (i,0).
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 */
void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // TODO
}

/*
 * @param n             The size of the input dimensions.
 * @param local_A       The distributed matrix A.
 * @param local_b       The distirbuted input vector b, distirbuted among the
 *                      first column of the processor grid (i,0).
 * @param local_x       The distributed output vector x, distributed among the
 *                      first column of the processor grid (i,0).
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 * @param max_iter          The maximum number of iterations to run.
 * @param l2_termination    The termination criteria for the L2-norm of
 *                          ||Ax - b||. Terminates as soon as the total L2-norm
 *                          is smaller or equal to this.
 */
// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    int rank;
	int coordinates[2];
	MPI_Comm_rank(comm, &rank);
	MPI_Cart_coords(comm, rank, 2, coordinates);
	
	int dims[2];
	int period[2];
	int coords[2];
	MPI_Cart_get(comm,2,dims,period,coords);
	
	int rankRoot;	
	int rankRowRoot;
	int rankRootDiag;
	int coordsRoot[2] = {0,0};
	int coordsRowRoot[2] = {coordinates[0], 0};
	int coordsRootDiag[2] = {coordinates[0], coordinates[0]};
	MPI_Cart_rank(comm, coordsRoot, &rankRoot);
	MPI_Cart_rank(comm, coordsRowRoot, &rankRowRoot);
	MPI_Cart_rank(comm, coordsRootDiag, &rankRootDiag);
	
	int rows = block_decompose(n, dims[0], coordinates[0]);
	
	double* diagonal = NULL;
	if(coordinates[1] == 0) {
		diagonal = new double[rows];
		
		if (rank != rankRootDiag) {
			for(int i=0; i < rows; i++) {
				MPI_Recv(diagonal + i, 1, MPI_DOUBLE, rankRootDiag, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
			}
		}
		else {
			for (int i=0; i < rows; i++) {
				diagonal[i] = local_A[i * rows + 1];
			}
		}
		
		for (int i = 0; i < rows; i++) { 
			local_x[i] = 0;
		}
	}
	else if (rank == rankRootDiag) {
		for (int i = 0; i < rows; i++) {
			MPI_Send((local_A + i * rows + i), 1, MPI_DOUBLE, rankRowRoot, 1, comm);
		}
	}
	
	double* resultAx = NULL;
	if (coordinates[1] == 0) {
		resultAx = new double[rows];
	}
	
	MPI_Comm colComm;
	int remain_dims[2] = {true, false};
	MPI_Cart_sub(comm, remain_dims, &colComm);
	
	MPI_Group groupCart, groupCol;
	MPI_Comm_group(comm, &groupCart);
	MPI_Comm_group(colComm, &groupCol);
	
	int rankCommRoot;
	MPI_Group_translate_ranks(groupCart, 1, &rankRoot, groupCol, &rankCommRoot);
	
	double resultLocal;
	double residual;
	int iterations = 0;
	
	while(1) {
		distributed_matrix_vector_mult(n, local_A, local_x, resultAx, comm);
		
		if (coordinates[1] == 0) {
			resultLocal = 0;
			for(int i = 0; i < rows; i++) {
				resultLocal = resultLocal + (resultAx[i] - local_b[i]) * (resultAx[i] - local_b[i]);
			}
			MPI_Reduce(&resultLocal, &residual, 1, MPI_DOUBLE, MPI_SUM, rankCommRoot, colComm);
		}
		
		if (rank == rankRoot) {
			residual = sqrt(residual);
		}
		
		MPI_Bcast(&residual, 1, MPI_DOUBLE, rankRoot, comm);
		
		if (iterations < max_iter) {
			if (residual < l2_termination) {
				std::cout<<"Iteration meets L2 termination condition"<<std::endl;
				break;
			}
			else {
				if (coordinates[1] == 0) {
					for (int i=0; i < rows; i++) {
						local_x[i] = (local_b[i] - resultAx[i] + diagonal[i]*local_x[i]) / diagonal[i];
					}
				}
				iterations++;
			}
		}
		else if(residual < l2_termination) {
			std::cout<<"Iteration meets L2 termination condition"<<std::endl;
			break;
		}
		else {
			std::cout<<"Reached max_iter without converging"<<std::endl;
			break;
		}
	}
	
	delete resultAx;
	delete diagonal;
	MPI_Comm_free(&colComm);
	MPI_Group_free(&groupCart);
	MPI_Group_free(&groupCol);
	
	return;
	

}

/*
 * @param n     The size of the dimensions.
 * @param A     A n-by-n matrix represented in row-major order.
 * @param x     The input vector of length n.
 * @param y     The output vector of length n.
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 */
 // wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

/*
 * @param n                 The size of the input.
 * @param A                 The input matrix `A` of size n-by-n.
 * @param b                 The input vector `b` of size n.
 * @param x                 The output vector `x` of size n.
 * @param comm              A 2d cartesian grid communicator of size q-by-q
 *                          (i.e., a perfect square).
 * @param max_iter          The maximum number of iterations to run.
 * @param l2_termination    The termination criteria for the L2-norm of
 *                          ||Ax - b||. Terminates as soon as the total L2-norm
 *                          is smaller or equal to this.
 */
// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
