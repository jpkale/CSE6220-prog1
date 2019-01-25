#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int dboard(int n) {
  return 3;
}

int main(int argc, char *argv[]) {
  int rank, p, n, r_per_process, r, m, grand_sum, i;
  double pi, time;

  /* Init MPI */
  MPI_Init(&argc, &argv);

  /* Get our rank */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Get our comm size */
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* Seed rand with rank */
  srand(rank);

  /* If we are in rank 0, parse arguments from command line */
  if (rank == 0) {
    if (argc != 3 || sscanf(argv[1], "%d", &n) != 1 || sscanf(argv[2], "%d", &r) != 1) {
      fprintf(stderr, "usage: ./prog1 N R\n");
      MPI_Finalize();
      return -1;
    }
  }

  /* Divide r equally among all of the processors */
  r_per_process = r/p;
  if (m % p && rank < (m % p)) {
    r_per_process++; 
  }

  /* Broadcast to the rest of the world */
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Start time */

  /* Call dboard r_per_process times in each process */
  m = 0;
  for (i = 0; i < r_per_process; i++) {
    m += dboard(n);
  }

  /* Reduce the sum into the root */
  MPI_Reduce(&m, &grand_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  /* Calculate pi */
  pi = 1.0*grand_sum;

  /* Rank 0 outputs the results */
  if (rank == 0) {
    printf("N=%d, R=%d, P=%d, PI=%lf\n", n, r, p, pi);
    printf("Time=%lf\n", time);
  }

  /* Finalize and quit */
  MPI_Finalize();
  return 0;
}
