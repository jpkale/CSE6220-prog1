#include <mpi.h>
#include <tuple>
#include <iostream>

#include "solver.h"

/******************** DECLARE YOUR HELPER FUNCTIONS HERE *******************/

using namespace std;

static const unsigned int KILL_SIGNAL = 0xffffffff;

/* Returns true if the first k columns in the solution are valid, and false
 * otherwise */
bool is_valid_partial_sol(const vector<unsigned int>& sol, unsigned int k) {
    /* If k is greater than the partial solution, this is invalid */
    if (k > sol.size()) {
        return false;
    }

    /* Go through every column in solution and ensure same row value for this
     * column does not appear anywhere else.  If it does, return false */
    for (unsigned int i=0; i<(k-1); i++) {
        for (unsigned int j=(i+1); j<k; j++) {
            if (sol[i] == sol[j]) {
                return false;
            }
        }
    }

    /* Found no duplicates, return true */
    return true;
}

/* SolutionTree class, used for generating new partial solutions */
class SolutionTree { 
public:
    SolutionTree(unsigned int n, unsigned int k);
    tuple<bool, vector<unsigned int>> next_partial_sol();
    unsigned int n;
    unsigned int k;

private:
    SolutionTree(unsigned int n, unsigned int k, unsigned int value,
            vector<unsigned int> prevs);
    vector<SolutionTree> children;
    bool is_root;
    unsigned int value;
};

SolutionTree::SolutionTree(unsigned int _n,
                           unsigned int _k,
                           unsigned int _value,
                           vector<unsigned int> prevs) {
    is_root = false;
    value = _value;
    n = _n;
    k = _k;

    prevs.push_back(value);

    /* Base case: no children to add */
    if (k == 0) {
        cout << "k=0 basecase" << endl;
        return;
    }

    /* Base case: no unique children to add */
    if (prevs.size() >= n) { return; }

    for (unsigned int i=0; i<n; i++) {
        if (find(prevs.begin(), prevs.end(), i) != prevs.end()) {
            cout << "recurse" << endl;
            children.push_back(SolutionTree(n, k-1, i, prevs));
        }
    }
}

SolutionTree::SolutionTree(unsigned int _n, unsigned int _k) {
    is_root = true;
    n = _n;
    k = _k;

    for (unsigned int i=0; i<n; i++) {
        children.push_back(SolutionTree(n, k-1, i, {}));
    }
}

tuple<bool, vector<unsigned int>> SolutionTree::next_partial_sol() {
    // TODO: Fix for 0-size board (n) or k=0

    /* Base case: no children */
    if (k == 0) {
        return make_tuple(true, vector<unsigned int>());
    }

    while (!children.empty()) {

        vector<unsigned int> child_sols;
        bool child_sols_valid;
        tie(child_sols_valid, child_sols) = children.back().next_partial_sol();

        if (child_sols_valid) {
            if (!is_root) {
                child_sols.push_back(value);
            }
            cout << "returning valid" << endl;
            return make_tuple(true, child_sols);
        }
        else {
            children.pop_back();
        }
    }

    cout << "returning invalid" << endl;
    return make_tuple(false, vector<unsigned int>());
}

static SolutionTree *sol_tree;

/* Given a board size n and partial solution size k, generate a solution to the
 * first k columns of the n-queens problem.  This solution is unique for each
 * call.  When no more unique solutions are available, the boolean in the tuple
 * is false, and the vector is unspecified */
tuple<bool, vector<unsigned int>> partial_sol(unsigned int n, unsigned int k) {
    if (!sol_tree) {
        sol_tree = new SolutionTree(n, k);
    }
    else if (n != sol_tree->n || k != sol_tree->k) {
        delete sol_tree;
        sol_tree = new SolutionTree(n, k);
    }

    cout << "partial_sol("<<n<<", "<<k<<");"<<endl;
    return sol_tree->next_partial_sol();
}

/* Create a set of complete solutions from a first-k-column partial solution
 * for a n-by-n size board */
vector<vector<unsigned int>> complete_sols(vector<unsigned int>& partial_sol,
                                           unsigned int k,
                                           unsigned int n) {
    /* Base case, k = n */
    if (k >= n) { return { partial_sol }; }

    /* Create vector of all our complete solutions to return */
    vector<vector<unsigned int>> all_sols;

    /* Try new row values for partial_sol[k] */
    for (unsigned int row=0; row<n; row++) {
        partial_sol.push_back(row);

        /* If the new row value we added created a valid partial solution, get
         * all complete solutions from this new partial solution, and append
         * them to our all_sols vector */
        if (is_valid_partial_sol(partial_sol, k+1)) {

            /* Recursively generate complete_sols for new partial_sol */
            auto curr_sols = complete_sols(partial_sol, k+1, n);

            /* Append these complete_sols to the end of our all_sols list */
            all_sols.insert(all_sols.end(), curr_sols.begin(),
                    curr_sols.end());
        }
    }

    /* Return all the complete solutions we found */
    return all_sols;
}


/*************************** solver.h functions ************************/

void seq_solver(unsigned int n,
                vector<vector<unsigned int>>& all_solns) {
	// TODO: Implement this function
}

void nqueen_master(unsigned int n,
                   unsigned int k,
                   vector<vector<unsigned int> >& all_solns) {
	/* Following is a general high level layout that you can follow
	 (you are not obligated to design your solution in this manner.
	  This is provided just for your ease). */ 

	/******************* STEP 1: Send one partial solution to each worker ********************/
	/*
	 * for (all workers) {
	 * 		- create a partial solution.
	 * 		- send that partial solution to a worker
	 * }
	 */

    /* Get number of processors (and workers) in world */
    int num_procs;
    unsigned int num_workers;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    num_workers = num_procs-1;

    /* Create request and status vectors */
    vector<MPI_Request> reqs(num_workers);

    /* Create a vector keeping track of which workers have terminated */
    vector<bool> has_termed(num_workers);

    /* Generate partial solution for each processor and use async-send to send
     * it to the destination */
    for (unsigned int worker=0; worker<num_workers; worker++) {
        vector<unsigned int> ps;
        bool ps_valid;
        tie(ps_valid, ps) = partial_sol(n, k);

        if (ps_valid) {
            cout << "ps_valid: " << ps_valid << endl;
            cout << "ps.size(): " << ps.size() << endl;
            cout << "ps[0]: " << ps[0] << endl;
            cout << "ps[1]: " << ps[1] << endl;
            cout << "About to isend partial solution" << endl;
            /* The partial solution generated was unique and valid */
            MPI_Isend(&ps[0], k, MPI_INT, worker+1, 0, MPI_COMM_WORLD,
                    &reqs[worker]);

            /* This worker has yet to terminate */
            has_termed[worker] = false;
        }
        else {
            /* The partial solution generated was non-unique, send term */
            MPI_Isend(&KILL_SIGNAL, 1, MPI_INT, worker+1, 1, MPI_COMM_WORLD,
                    &reqs[worker]);

            /* This worker has terminated */
            has_termed[worker] = true;
        }
    }

    /* Wait for all of the sends to go through */
    MPI_Waitall(num_workers, &reqs[0], MPI_STATUS_IGNORE);

	/******************* STEP 2: Send partial solutions to workers as they respond ********************/
	/*
	 * while() {
	 * 		- receive completed work from a worker processor.
	 * 		- create a partial solution
	 * 		- send that partial solution to the worker that responded
	 * 		- Break when no more partial solutions exist and all workers have responded with jobs handed to them
	 * }
	 */

    /* Create a vector that holds the number of solutions found by each worker */
    vector<unsigned int> num_sols_found(num_workers);

    /* Which worker has finished */
    int finished_worker;

    /* Get the total number of solutions each worker has generated with an
     * async recv call */
    for (unsigned int worker=0; worker<num_workers; worker++) {
        MPI_Irecv(&num_sols_found[worker], 1, MPI_INT, worker, 0,
                MPI_COMM_WORLD, &reqs[worker]);
    }

    /* Continue in while loop while there is some worker process that has not
     * terminated */
    while (find(has_termed.begin(), has_termed.end(), true) !=
            has_termed.end()) {
        /* Create a new partial solution */
        vector<unsigned int> ps;
        bool ps_valid;
        tie(ps_valid, ps) = partial_sol(n, k);

        /* Wait for any worker process to complete */
        MPI_Waitany(num_workers, &reqs[0], &finished_worker,
                MPI_STATUS_IGNORE);

        /* Set m_num_sols as the number of solutions found for this worker */
        unsigned int m_num_sols = num_sols_found[finished_worker];

        /* Create a 2D array for solutions */
        unsigned int *raw_worker_sols = (unsigned int *)malloc(n * m_num_sols);

        /* Read in all of the solutions found by this processor */
        MPI_Recv(raw_worker_sols, n*m_num_sols, MPI_INT, finished_worker+1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Demarshal raw solutions into vector, and place into all_solns */
        for (unsigned int i=0; i<m_num_sols; i++) {
            vector<unsigned int> temp(&raw_worker_sols[i*n*m_num_sols],
                    &raw_worker_sols[(i+1)*n*m_num_sols]);
            all_solns.push_back(temp);
        }

        if (ps_valid) {
            /* Partial solution is valid */ 

            /* Send the new partial solution to the processor which has
             * finished */
            MPI_Send(&ps[0], k, MPI_INT, finished_worker+1, 0, MPI_COMM_WORLD);
        }
        else {
            /* Partial does not exist, terminate the processor */
            MPI_Send(&KILL_SIGNAL, 1, MPI_INT, finished_worker+1, 1,
                    MPI_COMM_WORLD);
        }
    }

	/********************** STEP 3: Terminate **************************
	 *
	 * Send a termination/kill signal to all workers.
	 *
	 */
}

void nqueen_worker(unsigned int n,
                   unsigned int k) {

	// TODO: Implement this function

	// Following is a general high level layout that you can follow (you are not obligated to design your solution in this manner. This is provided just for your ease).

	/*******************************************************************
	 *
	 * while() {
	 *
	 * 		wait for a message from master
	 *
	 * 		if (message is a partial job) {
	 *				- finish the partial solution
	 *				- send all found solutions to master
	 * 		}
	 *
	 * 		if (message is a kill signal) {
	 *
	 * 				quit
	 *
	 * 		}
	 *	}
	 */

    /* vector of requests so we can use MPI_Waitany */
    vector<MPI_Request> reqs(2);

    /* Partial solution to receive */
    vector<unsigned int> ps(k);

    /* Termination signal space */
    unsigned int term_sig;

    /* Index of message received from wait any.  0=partial solution, 1=term
     * signal */
    int recvd_msg;

    /* Receive both a partial solution and a termination signal from the master
     * asynchronously */
    MPI_Irecv(&ps[0], k, MPI_INT, 0, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&term_sig, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &reqs[1]);

    /* Loop forever */
    while (true) {

        /* Wait for either the partial solution or the term signal */
        MPI_Waitany(2, &reqs[0], &recvd_msg, MPI_STATUS_IGNORE);

        if (recvd_msg == 0) {
            /* Received a partial solution */
            auto sols = complete_sols(ps, k, n);   

            /* Create a 2D array for solutions */
            unsigned int *raw_worker_sols = (unsigned int *)malloc(n *
                    sols.size());

            /* Fill raw solutions with values from complete solutions */
            for (unsigned int i=0; i<sols.size(); i++) {
                memcpy(&raw_worker_sols[i*n], &sols[i][0], n);
            }

            /* Send the 2D array to the master */
            MPI_Send(raw_worker_sols, n*sols.size(), MPI_INT, 0, 0,
                    MPI_COMM_WORLD);
            cout << "Send result from worker" << endl;

            /* Set up asynchronous receive again */
            MPI_Irecv(&ps[0], k, MPI_INT, 0, 0, MPI_COMM_WORLD, &reqs[0]);
        }
        else if (term_sig == KILL_SIGNAL) {
            /* Received a term signal */
            return;
        }
    }
}
