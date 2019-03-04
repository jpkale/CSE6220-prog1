#include "solver.h"


/*************************** DECLARE YOUR HELPER FUNCTIONS HERE ************************/
void addToAllSolutionSet(std::vector<std::vector<unsigned int> >& all_solns, std::vector<unsigned int> currentSolution);

void solveNQueens(int n, std::vector<std::vector<unsigned int>> board, int column, std::vector<std::vector<unsigned int> >& all_solns, std::vector<unsigned int> currentSolution);

bool spaceCheck(int n, std::vector<std::vector<unsigned int>> board, int row, int column);

void printBoard(int n, std::vector<std::vector<unsigned int>> board) ;

/*************************** solver.h functions ************************/

/************ Sequential solving function for nqueen*****************
 *
 * Takes the board size as input and creates all solutions to the n-queen problem. All solutions are put in the vector all_solns. Each solution is a vector of length n. Position i in a solution represents the row number of queen in column i. Columns and rows are numbered from 0 to n-1.
 *
 * Parameters:
 * n : The board size
 * all_solns: A vector of all solutions, each solution being an n sized vector.
 *
 * *****************************************************************/
void seq_solver(unsigned int n, std::vector<std::vector<unsigned int> >& all_solns) {
	
	std::vector<std::vector<unsigned int>> board(n, std::vector<unsigned int>(n));
	int currentColumn = 0;
	std::vector<unsigned int> currentSolution(n);
	
	solveNQueens(n, board, currentColumn, all_solns, currentSolution);
	
	
	// TODO: Implement this function



}






void nqueen_master(	unsigned int n,
					unsigned int k,
					std::vector<std::vector<unsigned int> >& all_solns) {




	// TODO: Implement this function

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


	/******************* STEP 2: Send partial solutions to workers as they respond ********************/
	/*
	 * while() {
	 * 		- receive completed work from a worker processor.
	 * 		- create a partial solution
	 * 		- send that partial solution to the worker that responded
	 * 		- Break when no more partial solutions exist and all workers have responded with jobs handed to them
	 * }
	 */

	/********************** STEP 3: Terminate **************************
	 *
	 * Send a termination/kill signal to all workers.
	 *
	 */





}

void nqueen_worker(	unsigned int n,
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


}



/*************************** DEFINE YOUR HELPER FUNCTIONS HERE ************************/

void addToAllSolutionSet(std::vector<std::vector<unsigned int> >& all_solns, std::vector<unsigned int> currentSolution) {
	all_solns.push_back(currentSolution);
}

void printBoard(int n, std::vector<std::vector<unsigned int>> board) 
{ 
    static int k = 1; 
    printf("%d-\n",k++); 
    for (int i = 0; i < n; i++) 
    { 
        for (int j = 0; j < n; j++) 
            printf(" %d ", board[i][j]); 
        printf("\n"); 
    } 
    printf("\n"); 
} 


void solveNQueens(int n, std::vector<std::vector<unsigned int>> board, int column, std::vector<std::vector<unsigned int> >& all_solns, std::vector<unsigned int> currentSolution) {
	
	if (column == n) { //All queens placed
		addToAllSolutionSet(all_solns, currentSolution);
		//printBoard(n, board);
		return;
	}
	
	for (int i=0; i<n; i++) {
		
		if (spaceCheck(n, board, i, column)) { //Check if space [i][column] is valid
			board[i][column] = 1;
			currentSolution[column] = i;
			
			solveNQueens(n, board, column+1, all_solns, currentSolution);
			
			board[i][column] = 0; //backtrack
			currentSolution[column] = 0;
		}
	}
	return;
}

bool spaceCheck(int n, std::vector<std::vector<unsigned int>> board, int row, int column) {

	// Check row
	for (int i=0; i<column; i++)
		if (board[row][i])
			return false;
		
	// Check up diagonal
	for (int i=row, j=column; i>=0 && j>= 0; i--, j--)
		if (board[i][j])
			return false;
		
	// Check low diagonal
	for (int i=row, j=column; j>=0 && i<n; i++, j--)
		if (board[i][j])
			return false;
	
	return true;
}





