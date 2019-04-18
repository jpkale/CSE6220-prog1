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

static MPI_Comm create_comm_by_dim(MPI_Comm comm, int dim)
{
    MPI_Comm new_comm;
    int remain_dims[NDIMS];

    for (int i=0; i<NDIMS; i++) {
        if (i == dim) {
            remain_dims[i] = false;
        } else {
            remain_dims[i] = true;
        }
    }

    MPI_Cart_sub(comm, remain_dims, &new_comm);
    return new_comm;
}

MPI_Comm create_col_comm(MPI_Comm comm)
{
    return create_comm_by_dim(comm, COL);
}

MPI_Comm create_row_comm(MPI_Comm comm)
{
    return create_comm_by_dim(comm, ROW);
}

int get_cart_rank(MPI_Comm comm, int row, int col)
{
    int coords[NDIMS];
    int rank;

    coords[ROW] = row;
    coords[COL] = col;
    MPI_Cart_rank(comm, coords, &rank);

    return rank;
}

static int translate_cart_to_1d_rank(MPI_Comm cart_comm, MPI_Comm oned_comm, int cart_rank)
{
    MPI_Group cart_group, oned_group;
    int oned_rank;

    MPI_Comm_group(cart_comm, &cart_group);
    MPI_Comm_group(oned_comm, &oned_group);

    MPI_Group_translate_ranks(cart_group, 1, &cart_rank, oned_group, &oned_rank);

    MPI_Group_free(&cart_group);
    MPI_Group_free(&oned_group);

    return oned_rank;
}

int translate_cart_to_col_rank(MPI_Comm cart_comm, MPI_Comm col_comm, int cart_rank)
{
    return translate_cart_to_1d_rank(cart_comm, col_comm, cart_rank);
}

int translate_cart_to_row_rank(MPI_Comm cart_comm, MPI_Comm row_comm, int cart_rank)
{
    return translate_cart_to_1d_rank(cart_comm, row_comm, cart_rank);
}

