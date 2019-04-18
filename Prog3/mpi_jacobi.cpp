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
#include <iostream>
#include <string.h>

#define DEBUG false

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
    int row = get_row(comm);
    int col = get_col(comm);

    //Create Column Communicator subset group for scattering vector
    MPI_Comm col_comm = create_col_comm(comm);


    //If not in the first column of processor grid, do nothing
    if (col != 0)
    {
        MPI_Comm_free(&col_comm);
        return;
    }

    //Calculate num of elements for each processor
    int num_rows = get_num_rows(comm);

    int* count = new int[num_rows];
    count[0] = block_decompose(n,num_rows,0);

    int* displs = new int[num_rows];
    displs[0] = 0;

    for (int i=1; i < num_rows; i++) {
        count[i] = block_decompose(n,num_rows,i);
        displs[i] = displs[i-1] + count[i-1];
    }

    //Get root rank and coordinates
    int rank_root = get_cart_rank(comm, 0, 0);

    //Scatter
    int size = block_decompose(n, num_rows, row);
    (*local_vector) = new double[size];
    int translation_rank = translate_cart_to_col_rank(comm, col_comm, rank_root);
    MPI_Scatterv(input_vector, count, displs, MPI_DOUBLE, *local_vector, size,
                 MPI_DOUBLE, translation_rank, col_comm);


    MPI_Comm_free(&col_comm);
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
    //Get rank and coordinates
    int rank = get_rank(comm);
    int row = get_row(comm);
    int col = get_col(comm);

    //Create Column Communicator subset group for scattering vector
    MPI_Comm col_comm = create_col_comm(comm);

    //If not in the first column of processor grid, do nothing
    if (col != 0)
    {
        MPI_Comm_free(&col_comm);
        return;
    }

    //Calculate num of elements for each processor
    int num_rows = get_num_rows(comm);

    int* count = NULL;
    int* displs = NULL;

    //Get root rank and coordinates
    int rank_root = get_cart_rank(comm, 0, 0);

    if (rank == rank_root) {
        count = new int[num_rows];
        displs = new int[num_rows];
        count[0] = block_decompose(n, num_rows, 0);
        displs[0] = 0;

        for (int i=0; i < num_rows; i++) {
            count[i] = block_decompose(n,num_rows,i);
            displs[i] = displs[i-1] + count[i-1];
        }
    }

    //Gather
    int size = block_decompose(n,num_rows, row);
    int translation_rank = translate_cart_to_col_rank(comm, col_comm, rank_root);
    MPI_Gatherv(local_vector, size, MPI_DOUBLE, output_vector, count, displs,
                MPI_DOUBLE, translation_rank, col_comm);

    MPI_Comm_free(&col_comm);
    delete count;
    delete displs;
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
    /* Get our rank, row, column, and the cart-group's root's rank */
    int row = get_row(comm);
    int col = get_col(comm);
    int cart_root_rank = get_cart_rank(comm, 0, 0);

    /* Get the total number of rows and columns in the cartesian group.  They
     * should be the same, but we get both just to be sure */
    int cart_comm_cols = get_num_cols(comm);
    int cart_comm_rows = get_num_rows(comm);

    /* Create a communicator for the current column */
    MPI_Comm col_comm = create_col_comm(comm);

    /* Create a communicator for the current row */
    MPI_Comm row_comm = create_row_comm(comm);

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

        int comm_root_rank = translate_cart_to_col_rank(comm, col_comm, cart_root_rank);
        MPI_Scatterv(input_matrix, count, displs, MPI_DOUBLE, temp, size,
                     MPI_DOUBLE, comm_root_rank, col_comm);

        delete count;
        delete displs;
    }

    MPI_Barrier(comm);

    int rows = block_decompose(n, cart_comm_rows, row);
    int columns = block_decompose(n, cart_comm_cols, col);
    (*local_matrix) = new double[rows*columns];

    int rank_col_root = get_cart_rank(comm, row, 0);

    int* count = new int[cart_comm_rows];
    int* displs = new int[cart_comm_rows];
    displs[0] = 0;
    count[0] = block_decompose(n, cart_comm_rows, 0);

    for(int i=1; i < cart_comm_rows; i++) {
        count[i] = block_decompose(n, cart_comm_rows, i);
        displs[i] = displs[i-1] + count [i-1];
    }

    int comm_rank_row_root = translate_cart_to_row_rank(comm, row_comm, rank_col_root);

    /* Distribute matrix from processors in first column to all processors in
     * the same row */
    for (int i=0; i < rows; i++) {
        MPI_Scatterv((temp + i*n), count, displs, MPI_DOUBLE,
                     (*local_matrix + i*columns), columns, MPI_DOUBLE,
                     comm_rank_row_root, row_comm);
    }

    /* Clean-up */
    delete count;
    delete displs;
    delete temp;
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
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


    if (row != 0) {
        /* 0-th column processor */
        if (col == 0) {
            /* Get the rank of the diagonal processor */
            int diag_rank = get_cart_rank(comm, row, row);

            /* Send our part of the vector to the diagonal processor */
            MPI_Send(col_vector, vector_size, MPI_DOUBLE, diag_rank, 0, comm);
        }

        /* Diagonal processor */
        if (row == col) {
            /* Get the rank of the first processor in the row */
            int first_rank = get_cart_rank(comm, row, 0);

            /* Receive our part of the vector from the first processor in the row */
            MPI_Recv(row_vector, vector_size, MPI_DOUBLE, first_rank, 0, comm,
                    MPI_STATUS_IGNORE);
        }
    } else if (col == 0) {
        memcpy(row_vector, col_vector, vector_size * sizeof(double));
    }

    /* At this point, the diagonal column has the distributed vector in
     * col_vector */


    /* Create communicator for column */
    MPI_Comm col_comm = create_col_comm(comm);

    /* Get rank of diagonal processor in column */
    int diag_rank = get_cart_rank(comm, col, col);

    /* Get rank for diagonal processor within col_comm */
    int col_diag_rank = translate_cart_to_col_rank(comm, col_comm, diag_rank);

    /* Broadcast into new communicator */
    int bcast_vector_size = block_decompose_by_dim(n, comm, COL);
    MPI_Bcast(row_vector, bcast_vector_size, MPI_DOUBLE, col_diag_rank, col_comm);

    /* Cleanup */
    MPI_Comm_free(&col_comm);
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
    num_cols = block_decompose_by_dim(n, comm, COL);

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

    /* Create communicator for current row */
    MPI_Comm row_comm = create_row_comm(comm);

    /* Get the rank for the first column's processor in the row communicator */
    int cart_root_rank = get_cart_rank(comm, get_row(comm), 0);
    int row_root_rank = translate_cart_to_row_rank(comm, row_comm, cart_root_rank);

    /* Reduce all partial vectors to the first column's processors */
    MPI_Reduce(&partial_res[0], local_y, num_rows, MPI_DOUBLE, MPI_SUM, row_root_rank, row_comm);
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
    int rank = get_rank(comm);
    int row = get_row(comm);
    int col = get_col(comm);

    int num_rows = get_num_rows(comm);

    int rank_root = get_cart_rank(comm, 0, 0);
    int rank_row_root = get_cart_rank(comm, row, 0);
    int rank_row_diag = get_cart_rank(comm, row, row);

    int rows = block_decompose(n, num_rows, row);

    double* diagonal = NULL;
    if(col == 0) {
        diagonal = new double[rows];

        if (rank != rank_row_diag) {
            for(int i=0; i < rows; i++) {
                MPI_Recv(diagonal + i, 1, MPI_DOUBLE, rank_row_diag, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
            }
        }
        else {
            for (int i=0; i < rows; i++) {
                diagonal[i] = local_A[i * rows + i];
            }
        }

        for (int i = 0; i < rows; i++) {
            local_x[i] = 0;
        }
    }
    else if (rank == rank_row_diag) {
        for (int i = 0; i < rows; i++) {
            MPI_Send((local_A + i * rows + i), 1, MPI_DOUBLE, rank_row_root, 1, comm);
        }
    }

    double* resultAx = NULL;
    if (col == 0) {
        resultAx = new double[rows];
    }

    MPI_Comm col_comm = create_col_comm(comm);
    int rank_comm_root = translate_cart_to_col_rank(comm, col_comm, rank_root);

    double resultLocal;
    double residual;
    int iterations = 0;

    while(1) {
        distributed_matrix_vector_mult(n, local_A, local_x, resultAx, comm);

        if (col == 0) {
            resultLocal = 0;
            for(int i = 0; i < rows; i++) {
                resultLocal = resultLocal + (resultAx[i] - local_b[i]) * (resultAx[i] - local_b[i]);
            }
            MPI_Reduce(&resultLocal, &residual, 1, MPI_DOUBLE, MPI_SUM, rank_comm_root, col_comm);
        }

        if (rank == rank_root) {
            residual = sqrt(residual);
        }

        MPI_Bcast(&residual, 1, MPI_DOUBLE, rank_root, comm);

        if (iterations < max_iter) {
            if (residual < l2_termination) {
                if (DEBUG)
                    std::cout<<"Iteration meets L2 termination condition"<<std::endl;
                break;
            }
            else {
                if (col == 0) {
                    for (int i=0; i < rows; i++) {
                        local_x[i] = (local_b[i] - resultAx[i] + diagonal[i]*local_x[i]) / diagonal[i];
                    }
                }
                iterations++;
            }
        }
        else if(residual < l2_termination) {
            if (DEBUG)
                std::cout<<"Iteration meets L2 termination condition"<<std::endl;
            break;
        }
        else {
            if (DEBUG)
                std::cout<<"Reached max_iter without converging"<<std::endl;
            break;
        }
    }

    delete resultAx;
    delete diagonal;
    MPI_Comm_free(&col_comm);
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
