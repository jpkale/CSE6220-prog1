#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <mpi.h>

#define RADIUS (1.0)
#define DEGREES_MAX (360)

#define DEG2RAD(x) (((x)/DEGREES_MAX)*M_PI*2.0)
#define RANDD(min,max) ((((rand()*1.0)/RAND_MAX)*(max-min))+min)

/* Global variables rank and p */
static int rank;
static int p;

int dboard(int n) {
  int i, n_per_process, m = 0;
  double a, theta, x, y;

  /* Compute floor(n/p) or ceil(n/p) */
  n_per_process = n / p;
  if (rank < (n % p)) {
    n_per_process++;
  }

  for (i = 0; i < n_per_process; i++) {
    a = RANDD(0, pow(RADIUS, 2));
    theta = RANDD(0, DEGREES_MAX);

    x = sqrt(a) * cos(DEG2RAD(theta));
    y = sqrt(a) * sin(DEG2RAD(theta));

    if ((fabs(x) <= RADIUS/sqrt(2.0)) && (fabs(y) <= RADIUS/sqrt(2.0))) {
      m++;
    }
  }

  return m;
}

int main(int argc, char *argv[]) {
  int n, r, m_i, m, i;
  double pi, pi_sum, pi_avg, start_time, end_time;

  /* Init MPI */
  MPI_Init(&argc, &argv);

  /* Get rank and p */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* Rank 0 parses arguments from command line */
  if (rank == 0) {
    if (argc != 3 || sscanf(argv[1], "%d", &n) != 1 || sscanf(argv[2], "%d", &r) != 1) {
      fprintf(stderr, "usage: ./prog1 N R\n");
      MPI_Finalize();
      return -1;
    }
  }

  /* Broadcast n and r to the rest of the world */
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&r, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Seed rand with rank */
  srand(rank);

  /* Rank 0 starts the clock */
  if (rank == 0) {
    start_time = MPI_Wtime();
  }

  /* Calculate pi r times and add to pi_sum */
  pi_sum = 0.0;
  for (i = 0; i < r; i++) {
    m_i = dboard(n);
    MPI_Reduce(&m_i, &m, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Rank 0 calculates pi from m and adds it to pi_sum */
    if (rank == 0) {
      pi = (2.0*n)/m;
      pi_sum = pi_sum + pi;
    }
  }

  /* Rank 0 ends the clock */
  if (rank == 0) {
    end_time = MPI_Wtime();
  }

  /* Rank 0 outputs the results */
  if (rank == 0) {
    pi_avg = pi_sum/r;
    printf("N=%d, R=%d, P=%d, PI=%lf\n", n, r, p, pi_avg);
    printf("Time=%lf\n", end_time-start_time);
  }

  /* Finalize and quit */
  MPI_Finalize();
  return 0;
}
