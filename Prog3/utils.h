/**
 * @file    utils.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements common utility/helper functions.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

/*********************************************************************
 *             You can add new functions to this header.             *
 *********************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>

/*********************************************************************
 * DO NOT CHANGE THE FUNCTION SIGNATURE OF THE FOLLOWING 3 FUNCTIONS *
 *********************************************************************/

inline int block_decompose(const int n, const int p, const int rank)
{
    return n / p + ((rank < n % p) ? 1 : 0);
}

inline int block_decompose(const int n, MPI_Comm comm)
{
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    return block_decompose(n, p, rank);
}

inline int block_decompose_by_dim(const int n, MPI_Comm comm, int dim)
{
    // get dimensions
    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    return block_decompose(n, dims[dim], coords[dim]);
}


/*********************************************************************
 *                  DECLARE YOUR OWN FUNCTIONS HERE                  *
 *********************************************************************/

#define ROW 0
#define COL 1
#define NDIMS 2

/* Get the current row in a cartesian communicator */
int get_row(MPI_Comm comm);

/* Get the current column in a cartesian communicator */
int get_col(MPI_Comm comm);

/* Get the rank in a given communicator */
int get_rank(MPI_Comm comm);

/* Get the number of rows in a cartesian communicator */
int get_num_rows(MPI_Comm comm);

/* Get the number of calls in a cartesian communicator */
int get_num_cols(MPI_Comm comm);

/* Given a cartesian comm, create a communicator for the processor's column in
 * that communicator.  Must be freed with MPI_Comm_free */
MPI_Comm create_col_comm(MPI_Comm comm);

/* Given a cartesian comm, create a communicator for the processor's row in
 * that communicator.  Must be freed with MPI_Comm_free */
MPI_Comm create_row_comm(MPI_Comm comm);

/* Get the rank of a processor in the cartesian communicator comm at the
 * position (row,col) */
int get_cart_rank(MPI_Comm comm, int row, int col);

/* Translate the cartesian rank of a processor to the column rank, given a
 * cartesian and column communicator */
int translate_cart_to_col_rank(MPI_Comm cart_comm, MPI_Comm col_comm, int cart_rank);

/* Translate the cartesian rank of a processor to the row rank, given a
 * cartesian and row communicator */
int translate_cart_to_row_rank(MPI_Comm cart_comm, MPI_Comm row_comm, int cart_rank);

#endif // UTILS_H
