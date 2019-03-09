#include <mpi.h>
#include <tuple>
#include <iostream>

#include "solver.h"

/******************** DECLARE YOUR HELPER FUNCTIONS HERE *******************/

using namespace std;

/* Signal to send to worker processes which should terminate */
static const unsigned int KILL_SIGNAL = 0xffffffff;

/* Returns true if the first k columns in the solution are valid, and false
 * otherwise */
bool is_valid_partial_sol(const vector<unsigned int>& sol);

/* Returns true if the vector contains the key.  False otherwise */
template <class T>
static bool contains(const vector<T>& v, const T& key) {
   return find(v.begin(), v.end(), key) != v.end();
}

/* SolutionTree class, used for generating new partial solutions */
class SolutionTree { 
public:
    /* Public constructor of nxn board with k partial solutions */
    SolutionTree(unsigned int n, unsigned int k);
    /* Get the next unique partial solution, if it exists */
    tuple<bool, vector<unsigned int>> next_partial_sol();
    /* size of board */
    unsigned int n;
    /* partial solutions size */
    unsigned int k;

private:
    /* Get the next traversal, which is not necessarily a solution */
    tuple<bool, vector<unsigned int>> next_traversal();
    /* The children of any node in the tree */
    vector<SolutionTree> children;
    /* Current child iterating through */
    unsigned int curr_value;
};

/* Singleton solution tree */
static SolutionTree *sol_tree;

/* Given a board size n and partial solution size k, generate a solution to the
 * first k columns of the n-queens problem.  This solution is unique for each
 * call.  When no more unique solutions are available, the boolean in the tuple
 * is false, and the vector is unspecified */
tuple<bool, vector<unsigned int>> partial_sol(unsigned int n, unsigned int k);

/* Create a set of complete solutions from a first-k-column partial solution
 * for a n-by-n size board */
vector<vector<unsigned int>> complete_sols(vector<unsigned int> partial_sol,
                                           unsigned int n); 


/*************************** solver.h functions ************************/

void seq_solver(unsigned int n,
                vector<vector<unsigned int>>& all_solns) {
	// TODO: Implement this function
}

void nqueen_master(unsigned int n,
                   unsigned int k,
                   vector<vector<unsigned int> >& all_solns) {

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
        bool ps_present;
        tie(ps_present, ps) = partial_sol(n, k);

        if (ps_present) {
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

    /* Create a vector that holds the number of solutions found by each worker */
    vector<unsigned int> num_sols_found(num_workers);

    /* Which worker has finished */
    int finished_worker;

    /* Get the total number of solutions each worker has generated with an
     * async recv call */
    for (unsigned int worker=0; worker<num_workers; worker++) {
        MPI_Irecv(&num_sols_found[worker], 1, MPI_INT, worker+1, 0,
                MPI_COMM_WORLD, &reqs[worker]);
    }

    /* Continue in while loop while there is some worker process that has not
     * terminated */
    while (contains(has_termed, false)) {

        /* Create a new partial solution */
        vector<unsigned int> ps;
        bool ps_present;
        tie(ps_present, ps) = partial_sol(n, k);

        /* Wait for any worker process to complete */
        MPI_Status status;
        MPI_Waitany(num_workers, &reqs[0], &finished_worker,
                &status);

        /* Set m_num_sols as the number of solutions found for this worker */
        unsigned int m_num_sols = num_sols_found[finished_worker];

        /* Create a 2D array for solutions */
        unsigned int *raw_worker_sols = (unsigned int *)malloc(n * m_num_sols *
                sizeof(unsigned int));

        /* Read in all of the solutions found by this processor */
        MPI_Recv(raw_worker_sols, n*m_num_sols, MPI_INT, finished_worker+1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Demarshal raw solutions into vector, and place into all_solns */
        for (unsigned int i=0; i<m_num_sols; i++) {
            vector<unsigned int> temp(&raw_worker_sols[i*n*m_num_sols],
                    &raw_worker_sols[(i+1)*n*m_num_sols]);
            all_solns.push_back(temp);
        }

        /* Free raw solutions array */
        free(raw_worker_sols);

        if (ps_present) {
            /* Partial solution is valid */ 

            /* Send the new partial solution to the processor which has
             * finished */
            MPI_Send(&ps[0], k, MPI_INT, finished_worker+1, 0, MPI_COMM_WORLD);

            /* Set up asynchronous receive of new completed solution */
            MPI_Irecv(&num_sols_found[finished_worker], 1, MPI_INT,
                    finished_worker+1, 0, MPI_COMM_WORLD,
                    &reqs[finished_worker]);
        }
        else {
            /* Partial does not exist, terminate the processor */
            MPI_Send(&KILL_SIGNAL, 1, MPI_INT, finished_worker+1, 1,
                    MPI_COMM_WORLD);
            has_termed[finished_worker] = true;
        }
    }
}

void nqueen_worker(unsigned int n,
                   unsigned int k) {

    /* vector of requests so we can use MPI_Waitany */
    vector<MPI_Request> reqs(2);

    /* Partial solution to receive */
    vector<unsigned int> ps(k);

    /* Termination signal space */
    unsigned int term_sig;

    /* Index of message received from wait any.  0=partial solution, 1=term
     * signal */
    int recvd_msg = 0;

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
            auto sols = complete_sols(ps, n);   
            unsigned int num_sols_found = sols.size();

            /* Create a 2D array for solutions */
            unsigned int *raw_worker_sols = (unsigned int *)malloc(n *
                    sols.size());

            /* Fill raw solutions with values from complete solutions */
            for (unsigned int i=0; i<sols.size(); i++) {
                sols[i] = vector<unsigned int>(&raw_worker_sols[i*n], &raw_worker_sols[(i+1)*n]);
            }

            /* Send the number of solutions found */
            MPI_Send(&num_sols_found, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            /* Send the 2D array to the master */
            MPI_Send(raw_worker_sols, n*sols.size(), MPI_INT, 0, 0,
                    MPI_COMM_WORLD);

            /* Free raw woker solutions */
            free(raw_worker_sols);

            /* Set up asynchronous receive again */
            MPI_Irecv(&ps[0], k, MPI_INT, 0, 0, MPI_COMM_WORLD, &reqs[0]);
        }
        else if (term_sig == KILL_SIGNAL) {
            /* Received a term signal */
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return;
        }
    }
}

/******************** DECLARE YOUR HELPER FUNCTIONS HERE *******************/

bool is_valid_partial_sol(const vector<unsigned int>& sol) {
    /* If solution is an empty vector, return true */
    if (sol.empty()) { return true; }

    /* Go through every column in solution and ensure same row value for this
     * column does not appear anywhere else.  If it does, return false */
    for (unsigned int i=0; i<sol.size()-1; i++) {
        for (unsigned int j=i+1; j<sol.size(); j++) {
            if (sol[i] == sol[j]) {
                return false;
            }
        }
    }
    /* Found no duplicates, return true */
    return true;
}

SolutionTree::SolutionTree(unsigned int _n, unsigned int _k) {
    curr_value = 0;
    n = _n;
    k = _k;

    /* Base case: no children to add */
    if (k <= 0) {
        return;
    }

    /* Add n children to intermediate node */
    for (unsigned int i=0; i<n; i++) {
        children.push_back(SolutionTree(n, k-1));
    }
}

tuple<bool, vector<unsigned int>> SolutionTree::next_traversal() {
    /* Base case, leaf node */
    if (k == 0 && curr_value < n) {
        curr_value = n+1;
        return make_tuple(true, vector<unsigned int>());
    }

    /* While we still have children left to iterate through */
    while (curr_value < n) {

        /* Get the child's traversal */
        vector<unsigned int> child_trav;
        bool child_trav_present;
        tie(child_trav_present, child_trav) = children[curr_value].next_traversal();

        /* If there is no child traversal, move onto next child */
        if (!child_trav_present) {
            curr_value++;
        }
        else {
            if (child_trav.empty()) {
                /* If the child was a leaf-node, we still need to move onto the
                 * next child, but also add our current value to the traversal */
                child_trav.push_back(curr_value);
                curr_value++;
            }
            else {
                /* Just add our current value and return */
                child_trav.push_back(curr_value);
            }
            return make_tuple(true, child_trav);
        }
    }

    /* No valid traversals in any of our children?  Then there's no valid
     * traversals for us */
    return make_tuple(false, vector<unsigned int>());
}

tuple<bool, vector<unsigned int>> SolutionTree::next_partial_sol() {
    vector<unsigned int> trav;
    bool trav_present;

    do {
        /* Get the next unique traversal */
        tie(trav_present, trav) = next_traversal();

        /* If a new unique traversal is not present, a partial solution won't
         * be either */
        if (!trav_present) {
            return make_tuple(false, vector<unsigned int>()); 
        }

        /* Do this until we have a valid partial solution */
    } while (!is_valid_partial_sol(trav));

    /* Return our valid partial solution */
    return make_tuple(trav_present, trav);
}

tuple<bool, vector<unsigned int>> partial_sol(unsigned int n, unsigned int k) {

    /* Create the solution tree for this (n,k) pair if it does not already
     * exist */
    if (!sol_tree) {
        sol_tree = new SolutionTree(n, k);
    }
    else if (n != sol_tree->n || k != sol_tree->k) {
        delete sol_tree;
        sol_tree = new SolutionTree(n, k);
    }

    /* Return the next partial solution from the tree */
    return sol_tree->next_partial_sol();
}

vector<vector<unsigned int>> complete_sols(vector<unsigned int> partial_sol,
                                           unsigned int n) { 
    /* Base case, k >= n */
    if (partial_sol.size() >= n) { return { partial_sol }; }

    /* Create vector of all our complete solutions to return */
    vector<vector<unsigned int>> all_sols;

    /* Try new row values for partial_sol[k] */
    for (unsigned int row=0; row<n; row++) {

        /* If the new row value we added created a valid partial solution, get
         * all complete solutions from this new partial solution, and append
         * them to our all_sols vector */
        if (!contains(partial_sol, row)) {
            partial_sol.push_back(row);

            /* Recursively generate complete_sols for new partial_sol */
            auto curr_sols = complete_sols(partial_sol, n);

            /* Append these complete_sols to the end of our all_sols list */
            all_sols.insert(all_sols.end(), curr_sols.begin(),
                    curr_sols.end());
        }
    }

    /* Return all the complete solutions we found */
    return all_sols;
}
