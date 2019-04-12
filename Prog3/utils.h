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

/* TODO: Move the all to cpp file, memoize*/

inline int get_row(MPI_Comm comm)
{
    int dims[NDIMS], periods[NDIMS], coords[NDIMS];
    MPI_Cart_get(comm, NDIMS, dims, periods, coords);
    return coords[ROW];
}

inline int get_col(MPI_Comm comm)
{
    int dims[NDIMS], periods[NDIMS], coords[NDIMS];
    MPI_Cart_get(comm, NDIMS, dims, periods, coords);
    return coords[COL];
}

inline int get_rank(MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

inline int get_cart_root_rank(MPI_Comm comm)
{
    int rank, coords[NDIMS];
    coords[0] = 0;
    coords[1] = 0;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

inline int get_num_rows(MPI_Comm comm) {
    int dims[NDIMS], period[NDIMS], coords[NDIMS];
    MPI_Cart_get(comm, NDIMS, dims, period, coords);
    return dims[ROW];
}

inline int get_num_cols(MPI_Comm comm) {
    int dims[NDIMS], period[NDIMS], coords[NDIMS];
    MPI_Cart_get(comm, NDIMS, dims, period, coords);
    return dims[COL];
}

#endif // UTILS_H
