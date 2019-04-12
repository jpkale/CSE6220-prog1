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
	int rank = 0;
	int coordinates[NDIMS];
	
	//Get rank and coordinates
	MPI_Comm_rank(comm, &rank);
	MPI_Cart_coords(comm, rank, NDIMS, coordinates);
	
	//Create Column Communicator subset group for scattering vector
	MPI_Comm colComm;
	int remain_dims[NDIMS];
    remain_dims[ROW] = true;
    remain_dims[COL] = false;
	MPI_Cart_sub(comm, remain_dims, &colComm);
	
	// Create communication groups
	MPI_Group cartesianGroup, columnGroup;
	MPI_Comm_group(comm, &cartesianGroup);
	MPI_Comm_group(colComm, &columnGroup);
	
	//If not in the first column of processor grid, do nothing
	if (coordinates[COL] != 0)
	{
		MPI_Group_free(&cartesianGroup);
		MPI_Group_free(&columnGroup);
		MPI_Comm_free(&colComm);
		return;
	}
	
	//Calculate num of elements for each processor
	int dims[NDIMS], periods[NDIMS], coords[NDIMS];
	MPI_Cart_get(comm,NDIMS,dims,periods,coords);
	
	int* count = new int[dims[ROW]];
	count[0] = block_decompose(n,dims[ROW],0);
	
	int* displs = new int[dims[ROW]];
	displs[0] = 0;
	
	for (int i=1; i < dims[ROW]; i++) {
		count[i] = block_decompose(n,dims[ROW],i);
		displs[i] = displs[i-1] + count[i-1];
	}
	
	//Get root rank and coordinates
	int rankRoot;
	int coordRoot[] = {0,0};
	MPI_Cart_rank(comm, coordRoot, &rankRoot);

	//Scatter
	int size = block_decompose(n, dims[ROW], coordinates[ROW]);
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
	int rank = 0;
	int coordinates[NDIMS];
	
	//Get rank and coordinates
	MPI_Comm_rank(comm, &rank);
	MPI_Cart_coords(comm, rank, NDIMS, coordinates);
	
	//Create Column Communicator subset group for scattering vector
	MPI_Comm colComm;
	int remain_dims[NDIMS];
    remain_dims[ROW] = true;
    remain_dims[COL] = false;
	MPI_Cart_sub(comm, remain_dims, &colComm);
	
	MPI_Group cartesianGroup, columnGroup;
	MPI_Comm_group(comm, &cartesianGroup);
	MPI_Comm_group(comm, &columnGroup);
	
	//If not in the first column of processor grid, do nothing
	if (coordinates[COL] != 0)
	{
		MPI_Group_free(&cartesianGroup);
		MPI_Group_free(&columnGroup);
		MPI_Comm_free(&colComm);
		return;
	}
	
	//Calculate num of elements for each processor
	int dims[NDIMS], periods[NDIMS], coords[NDIMS];
	MPI_Cart_get(comm,NDIMS,dims,periods,coords);
	
	int* count = NULL;
	int* displs = NULL;
	
	//Get root rank and coordinates
	int rankRoot;
	int coordRoot[] = {0,0};
	MPI_Cart_rank(comm, coordRoot, &rankRoot);
	
	if (rank == rankRoot) {
		count = new int[dims[ROW]];
		displs = new int[dims[ROW]];
		count[0] = block_decompose(n, dims[ROW], 0);
		displs[0] = 0;
		
		for (int i=0; i < dims[ROW]; i++) {
			count[i] = block_decompose(n,dims[ROW],i);
			displs[i] = displs[i-1] + count[i-1];
		}
	}
	
	//Gather
	int size = block_decompose(n,dims[ROW], coordinates[ROW]);
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
    int rank, row, col, cart_root_rank;

    /* Get our rank, row, column, and the cart-group's root's rank */
    rank = get_rank(comm);
    row = get_row(comm);
    col = get_col(comm);
    cart_root_rank = get_cart_root_rank(comm);

    /* Get the total number of rows and columns in the cartesian group.  They
     * should be the same, but we get both just to be sure */
    int cart_comm_rows, cart_comm_cols;
    cart_comm_cols = get_num_cols(comm);
    cart_comm_rows = get_num_rows(comm);

    /* Create a communicator for the current column */
	MPI_Comm colComm;
	int remain_dims[NDIMS];
    remain_dims[ROW] = true;
    remain_dims[COL] = false;
	MPI_Cart_sub(comm, remain_dims, &colComm);

    /* Create a communicator for the current row */
	MPI_Comm rowComm;
	remain_dims[ROW] = false;
	remain_dims[COL] = true;
	MPI_Cart_sub(comm, remain_dims, &rowComm);

    /* Get the groups for the current column, row, and cartesian communicators
     * for easy conversion between ranks later */
	MPI_Group groupCart, groupCol, groupRow;
	MPI_Comm_group(comm, &groupCart);
	MPI_Comm_group(colComm, &groupCol);
	MPI_Comm_group(rowComm, &groupRow);

    /* Distribute matrix among processors in first column */
	double* temp = NULL;
	if (col == 0) {
		int* count = new int[cart_comm_cols];
		int* displs = new int[cart_comm_cols];
		
		displs[0] = 0;
		count[0] = n * block_decompose(n, cart_comm_cols, 0);
		
		for (int i=1; i < cart_comm_cols; i++) {
			count[i] = n * block_decompose(n, cart_comm_cols, i);
			displs[i] = displs[i-1] + count[i-1];
		}
	
        int size = n * block_decompose(n, cart_comm_cols, col);
		temp = new double[size];

		int commRootRank;
		MPI_Group_translate_ranks(groupCart, 1, &cart_root_rank, groupCol, &commRootRank);
		MPI_Scatterv(input_matrix, count, displs, MPI_DOUBLE, temp, size, MPI_DOUBLE, commRootRank, colComm);
		
		delete count;
		delete displs;
	}

	MPI_Barrier(comm);
	
	int rows = block_decompose(n, cart_comm_rows, row);
	int columns = block_decompose(n, cart_comm_cols, col);
	(*local_matrix) = new double[rows*columns];
	
	int rankColRoot;
	int coordinatesColRoot[NDIMS];
    coordinatesColRoot[ROW] = row;
    coordinatesColRoot[COL] = 0;
	MPI_Cart_rank(comm, coordinatesColRoot,&rankColRoot);
	
	int* count = new int[cart_comm_rows];
	int* displs = new int[cart_comm_rows];
	displs[0] = 0;
	count[0] = block_decompose(n, cart_comm_rows, 0);
	
	for(int i=1; i < cart_comm_rows; i++) {
		count[i] = block_decompose(n, cart_comm_rows, i);
		displs[i] = displs[i-1] + count [i-1];
	}
	
	int commRankRowRoot;
	MPI_Group_translate_ranks(groupCart, 1, &rankColRoot, groupRow, &commRankRowRoot);

    /* Distribute matrix from processors in first column to all processors in
     * the same row */
	for (int i=0; i < rows; i++) {
		MPI_Scatterv((temp + i*n), count, displs, MPI_DOUBLE,
                     (*local_matrix + i*columns), columns, MPI_DOUBLE,
                     commRankRowRoot, rowComm);
	}

#if 0
    printf("In rank %d (%d,%d), local_matrix = [\n", get_rank(comm), get_row(comm), get_col(comm));
    for (int i=0; i<rows; i++) {
        for (int j=0; j<columns; j++) {
            printf(" %lf", (*local_matrix)[i*rows + j]);
        }
        printf("\n");
    }
    printf("]\n");

#endif /* 1 */

    /* Clean-up */
	delete count;
	delete displs;
	delete temp;
	MPI_Comm_free(&rowComm);
	MPI_Comm_free(&colComm);
	MPI_Group_free(&groupCart);
	MPI_Group_free(&groupRow);
	MPI_Group_free(&groupCol);
	return;
	
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
    int row, col, vector_size;

    row = get_row(comm);
    col = get_col(comm);

    /* Set vector_size to floor(n/q) or ceil(n/q) */
    vector_size = block_decompose_by_dim(n, comm, ROW);

    /* 0-th column processor */
    if (col == 0) {
        int diag_rank;
        int diag_coords[NDIMS];

        /* Populate diag_coords */
        diag_coords[ROW] = row;
        diag_coords[COL] = row;

        /* Get the rank of the diagonal processor */
        MPI_Cart_rank(comm, diag_coords, &diag_rank);

        /* Send our part of the vector to the diagonal processor */
        MPI_Send(col_vector, vector_size, MPI_DOUBLE, diag_rank, 0, comm);
    }

    /* Diagonal processor */
    if (row == col) {
        int first_coords[NDIMS];
        int first_rank;

        /* Populate first_coords */
        first_coords[ROW] = row;
        first_coords[COL] = 0;

        /* Get the rank of the first processor in the row */
        MPI_Cart_rank(comm, first_coords, &first_rank);

        /* Receive our part of the vector from the first processor in the row */
        MPI_Recv(row_vector, vector_size, MPI_DOUBLE, first_rank, 0, comm,
                MPI_STATUS_IGNORE);

#if 0
        printf("In rank %d (%d,%d), (before bcast) row_vector = [", get_rank(comm), get_row(comm), get_col(comm));
        for (int i=0; i<vector_size; i++) {
            printf(" %lf", row_vector[i]);
        }
        printf("]\n");
#endif
    }

    /* At this point, the diagonal column has the distributed vector in
     * col_vector */

    MPI_Comm col_comm;

    /* Create communicator for column */
    int remain_dims[NDIMS];
    remain_dims[ROW] = true;
    remain_dims[COL] = false;
    MPI_Cart_sub(comm, remain_dims, &col_comm);

    /* Get rank of diagonal processor in column */
    int diag_rank;
    int diag_coords[NDIMS];
    diag_coords[ROW] = col;
    diag_coords[COL] = col;
    MPI_Cart_rank(comm, diag_coords, &diag_rank);

    /* Get rank for diagonal processor within col_comm */
    int col_diag_rank;
    MPI_Group cart_group, col_group;
    MPI_Comm_group(comm, &cart_group);
    MPI_Comm_group(col_comm, &col_group);
    MPI_Group_translate_ranks(cart_group, 1, &diag_rank, col_group, &col_diag_rank);

    /* Broadcast into new communicator */
    MPI_Bcast(row_vector, vector_size, MPI_DOUBLE, col_diag_rank, col_comm);

#if 0
    printf("In rank %d (%d,%d), (after bcast) row_vector = [", get_rank(comm), get_row(comm), get_col(comm));
    for (int i=0; i<vector_size; i++) {
        printf(" %lf", row_vector[i]);
    }
    printf("]\n");
#endif

    /* Cleanup */
    MPI_Comm_free(&col_comm);
    MPI_Group_free(&cart_group);
    MPI_Group_free(&col_group);
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
    int num_rows, num_cols;
    std::vector<double> transposed_x(n);

    /* Transpose x and distribute */
    transpose_bcast_vector(n, local_x, &transposed_x[0], comm);

    /* Determine num_rows */
    num_rows = block_decompose_by_dim(n, comm, ROW);
    num_cols = block_decompose_by_dim(n, comm, ROW);

    /* Allocate new vector for partial result */
    std::vector<double> partial_res(num_rows);

    /* Initialize local_y = [0; 0; ... 0] */
    for (int row=0; row<num_rows; row++) {
        partial_res[row] = 0.0;
    }


    /* Calculate y = A*x, row-by-row */
    for (int row=0; row<num_rows; row++) {
        for (int col=0; col<num_cols; col++) {
            partial_res[row] += local_A[row * num_cols + col] * transposed_x[col];
        }
    }

#if 0
    printf("In rank %d (%d,%d), partial_res = [\n", get_rank(comm), get_row(comm), get_col(comm));
    for (int i=0; i<num_rows; i++) {
        printf(" %lf\n", partial_res[i]);
    }
    printf("]\n");

#endif

    /* Create communicator for current row */
    int remain_dims[NDIMS];
    MPI_Comm row_comm;
    remain_dims[ROW] = false;
    remain_dims[COL] = true;
    MPI_Cart_sub(comm, remain_dims, &row_comm);

    /* Get groups for both the cartesian and row communicators */
    MPI_Group cart_group, row_group;
    MPI_Comm_group(comm, &cart_group);
    MPI_Comm_group(row_comm, &row_group);

    /* Get the rank for the first column's processor in the row communicator */
    int row_group_root_rank, cart_group_root_rank;
    int row_root_coords[NDIMS];
    row_root_coords[ROW] = get_row(comm);
    row_root_coords[COL] = 0;
    MPI_Cart_rank(comm, row_root_coords, &cart_group_root_rank);
    MPI_Group_translate_ranks(cart_group, 1, &cart_group_root_rank, row_group, &row_group_root_rank);

    /* Reduce all partial vectors to the first column's processors */
    MPI_Reduce(&partial_res[0], local_y, num_rows, MPI_DOUBLE, MPI_SUM, row_group_root_rank, row_comm);
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
