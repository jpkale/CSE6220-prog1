#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <mpi.h>

#define RADIUS_MAX (1.0)
#define DEGREES_MAX (360)

#define DEG2RAD(x) (((x)/360.0)*M_PI*2.0)
#define RANDD(min,max) ((((rand()*1.0)/RAND_MAX)*(max-min))+min)

int dboard(int n) {
  int x, y, m, i;

  for (i=0; i<n; i++) {
    x = RANDD(-1.0, 1.0);
    y = RANDD(-1.0, 1.0);

    m += (sqrt(pow(x,2) + pow(y,2)) <= 1);
  }

  return m;
}

int main(int argc, char *argv[]) {
  int rank, p, n, n_per_process, m_per_process, m, r;
  double pi, time;

  /* Init MPI */
  MPI_Init(&argc, &argv);

  /* Get our rank */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Get our comm size */
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* If we are in rank 0, parse arguments from command line */
  if (rank == 0) {
    if (argc != 3 || sscanf(argv[1], "%d", &n) != 1 || sscanf(argv[2], "%d", &r) != 1) {
      fprintf(stderr, "usage: ./prog1 N R\n");
      MPI_Finalize();
      return -1;
    }
  }

  /* Broadcast n and r to the rest of the world */
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Divide n equally among all of the processors */
  n_per_process = n/p;
  if ((n % p) && rank < (n % p)) {
    n_per_process++; 
  }
  printf("In rank %d, n_per_process=%d\n", rank, n_per_process);

  /* Seed rand with rank */
  srand(rank);

  /* Start time */

  /* Call dboard in each process */
  m_per_process = dboard(n_per_process);

  printf("In rank %d, m_per_process=%d\n", rank, m_per_process);

  /* Reduce the sum into the root */
  MPI_Reduce(&m_per_process, &m, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  /* Average the results we got */
  pi = (4.0*m)/n;

  /* Rank 0 outputs the results */
  if (rank == 0) {
    printf("N=%d, R=%d, P=%d, PI=%lf\n", n, r, p, pi);
    printf("Time=%lf\n", time);
  }

  /* Finalize and quit */
  MPI_Finalize();
  return 0;
}
