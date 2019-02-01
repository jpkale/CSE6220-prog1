#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <mpi.h>

#define RADIUS_MAX (1.0)
#define DEGREES_MAX (360)

#define DEG2RAD(x) (((x)/360.0)*M_PI*2.0)
#define RANDD(min,max) ((((rand()*1.0)/RAND_MAX)*(max-min))+min)

#define DEBUG (true)

int dboard(int n) {
  int x, y, m, i, p = 0;
  
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  int nDarts = n/p;

  for (i=0; i<nDarts; i++) {
    x = RANDD(-1.0, 1.0);
    y = RANDD(-1.0, 1.0);

    m += (sqrt(pow(x,2) + pow(y,2)) <= 1);
  }

  return m;
}

int main(int argc, char *argv[]) {
  int rank, p, n, n_per_process, m_i, m, r = 0;
  double pi, time, pi_sum, pi_avg = 0.0;

  /* Init MPI */
  MPI_Init(&argc, &argv);
  
  double t0 = MPI_Wtime();

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
  MPI_Bcast(&r, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Divide n equally among all of the processors */
  //n_per_process = n/p;
  //if ((n % p) && rank < (n % p)) {
  //  n_per_process++; 
  //}
  //if DEBUG
//	printf("In rank %d, n_per_process=%d\n", rank, n_per_process);

  /* Seed rand with rank */
  srand(rank);

  /* Start time */

  /* Call dboard in each process */
  
  pi_sum = 0.0;
  for (int i=0; i<r; i++) {
	  //m_per_process = dboard(n_per_process);
	  m_i = dboard(n);
	  if DEBUG
		printf("In rank %d, m_per_process=%d, n=%d, p=%d\n", rank, m_i, n, p);

	  if (rank == 0) {
		  MPI_Reduce(&m_i, &m, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		  pi = (4.0*m)/n;
		  pi_sum = pi_sum + pi;
	  }
  }
  //m_per_process = dboard(n_per_process);


  /* Reduce the sum into the root */
  //MPI_Reduce(&m_per_process, &m, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  /* Average the results we got */
  //pi = (4.0*m)/n;

  /* Rank 0 outputs the results */
  double t1 = MPI_Wtime();
  if (rank == 0) {
	time = t1-t0;
    pi_avg = pi_sum/r;
	printf("N=%d, R=%d, P=%d, PI=%lf, PI_AVG=%lf\n", n, r, p, pi, pi_avg);
    printf("Time=%lf\n", time);
  }

  /* Finalize and quit */
  MPI_Finalize();
  return 0;
}
