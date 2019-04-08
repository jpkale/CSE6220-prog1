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
	int rank = 0;
	int coordinates[2];
	
	//Get rank and coordinates
	MPI_Comm_rank(comm, &rank);
	MPI_Cart_coords(comm, rank, 2, coordinates);
	
	//Create Column Communicator subset group for scattering vector
	MPI_Comm colComm;
	int remain_dims[2] = {true, false};
	MPI_Cart_sub(comm, remain_dims, &colComm);
	
	// Create communication groups
	MPI_Group cartesianGroup, columnGroup;
	MPI_Comm_group(comm, &cartesianGroup);
	MPI_Comm_group(comm, &columnGroup);
	
	//If not in the first column of processor grid, do nothing
	if (coordinates[1] != 0)
	{
		MPI_Group_free(&cartesianGroup);
		MPI_Group_free(&columnGroup);
		MPI_Comm_free(&colComm);
		return;
	}
	
	//Calculate num of elements for each processor
	int dims[2], periods[2], coords[2];
	MPI_Cart_get(comm,2,dims,periods,coords);
	
	int* count = new int[dims[0]];
	count[0] = block_decompose(n,dims[0],0);
	
	int* displs = new int[dims[0]];
	displs[0] = 0;
	
	for (int i=0; i < dims[0]; i++) {
		count[i] = block_decompose(n,dims[0],i);
		displs[i] = displs[i-1] + count[i-1];
	}
	
	//Get root rank and coordinates
	int rankRoot;
	int coordRoot[] = {0,0};
	MPI_Cart_rank(comm, coordRoot, &rankRoot);
	
	//Scatter
	int size = block_decompose(n,dims[0], coordinates[0]);
	(*local_vector) = new double[size];
	int translationRank;
	MPI_Group_translate_ranks(cartesianGroup, 1, &rankRoot, columnGroup, &translationRank);
	MPI_Scatterv(input_vector, count, displs, MPI_DOUBLE, *local_vector, size, MPI_DOUBLE, translationRank, colComm);
	
	MPI_Group_free(&cartesianGroup);
	MPI_Group_free(&columnGroup);
	MPI_Comm_free(&colComm);
	delete count;
	delete displs;
	return;

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
	int rank = 0;
	int coordinates[2];
	
	//Get rank and coordinates
	MPI_Comm_rank(comm, &rank);
	MPI_Cart_coords(comm, rank, 2, coordinates);
	
	//Create Column Communicator subset group for scattering vector
	MPI_Comm colComm;
	int remain_dims[2] = {true, false};
	MPI_Cart_sub(comm, remain_dims, &colComm);
	
	MPI_Group cartesianGroup, columnGroup;
	MPI_Comm_group(comm, &cartesianGroup);
	MPI_Comm_group(comm, &columnGroup);
	
	//If not in the first column of processor grid, do nothing
	if (coordinates[1] != 0)
	{
		MPI_Group_free(&cartesianGroup);
		MPI_Group_free(&columnGroup);
		MPI_Comm_free(&colComm);
		return;
	}
	
	//Calculate num of elements for each processor
	int dims[2], periods[2], coords[2];
	MPI_Cart_get(comm,2,dims,periods,coords);
	
	int* count = NULL;
	int* displs = NULL;
	
	//Get root rank and coordinates
	int rankRoot;
	int coordRoot[] = {0,0};
	MPI_Cart_rank(comm, coordRoot, &rankRoot);
	
	if (rank == rankRoot) {
		count = new int[dims[0]];
		displs = new int[dims[0]];
		count[0] = block_decompose(n, dims[0], 0);
		displs[0] = 0;
		
		for (int i=0; i < dims[0]; i++) {
			count[i] = block_decompose(n,dims[0],i);
			displs[i] = displs[i-1] + count[i-1];
		}
	}
	
	//Gather
	int size = block_decompose(n,dims[0], coordinates[0]);
	int translationRank;
	MPI_Group_translate_ranks(cartesianGroup, 1, &rankRoot, columnGroup, &translationRank);
	MPI_Gatherv(local_vector, size, MPI_DOUBLE, output_vector, count, displs, MPI_DOUBLE, translationRank, colComm);
	
	MPI_Group_free(&cartesianGroup);
	MPI_Group_free(&columnGroup);
	MPI_Comm_free(&colComm);
	delete count;
	delete displs;
	return;
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
    // TODO
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
