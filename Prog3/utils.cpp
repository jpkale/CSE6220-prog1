/**
 * @file    utils.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements common utility/helper functions.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "utils.h"
#include <iostream>

/*********************************************************************
 *                 Implement your own functions here                 *
 *********************************************************************/

int get_row(MPI_Comm comm)
{
    int dims[NDIMS], periods[NDIMS], coords[NDIMS];
    MPI_Cart_get(comm, NDIMS, dims, periods, coords);
    return coords[ROW];
}

int get_col(MPI_Comm comm)
{
    int dims[NDIMS], periods[NDIMS], coords[NDIMS];
    MPI_Cart_get(comm, NDIMS, dims, periods, coords);
    return coords[COL];
}

int get_rank(MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int get_cart_root_rank(MPI_Comm comm)
{
    int rank, coords[NDIMS];
    coords[0] = 0;
    coords[1] = 0;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

int get_num_rows(MPI_Comm comm)
{
    int dims[NDIMS], period[NDIMS], coords[NDIMS];
    MPI_Cart_get(comm, NDIMS, dims, period, coords);
    return dims[ROW];
}

int get_num_cols(MPI_Comm comm)
{
    int dims[NDIMS], period[NDIMS], coords[NDIMS];
    MPI_Cart_get(comm, NDIMS, dims, period, coords);
    return dims[COL];
}
