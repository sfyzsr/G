#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

// Initialize the row 
int n;

// Initialize augmented matrix with big capacity
double A[100][100];
// Initialize temp solution list for the random generator
double x[100];

// Initialize the process rank and the number of processes
int my_rank, comm_sz;

// Function header
void initializeMatrix();
void initializeRandomMatrix();
void gauss();
void backSubstitution();
void printMatrix();
void printSolution();

int main(int argc, char **argv) {

    struct timeval wallStart, wallEnd;

    // Initialize MPI
    MPI_Init(NULL, NULL);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // Get total process
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    printf("Use %d process\n", comm_sz);

    // Process 0 initialze matrix
    if(my_rank == 0 ){

        // Set the wall clock
        gettimeofday(&wallStart, NULL); 
       
        initializeRandomMatrix();
        printf("Initial argumented matrix: \n");
        printMatrix();
    }

    // Gaussian Elimination
    gauss();

    // Process 0 do the back substitution and print matrix
    if(my_rank == 0 ){

        backSubstitution();

        printMatrix();
        printSolution();

        gettimeofday(&wallEnd, NULL); // end time
        double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);
        double WALLtimeTaken = 1.0E-06 * ((wallSecs*1000000) + (wallEnd.tv_usec - wallStart.tv_usec)); // microseconds
        printf("Wall clock run Time: %f  seconds  \n", WALLtimeTaken);
    }

    // Finish MPI
    MPI_Finalize();
    return 0;
}

void initializeRandomMatrix() {
// Generate random matrix by the random seed

    // Since the total time is small, I use the decimal part of the double as random seed
    double start_time = MPI_Wtime();
    int seed = (int)((start_time - (int)start_time) * 1000000);
    // seed = 1;   // set 1 to test
    printf("seed = %d\n", seed);
    srand(seed);

    n = rand()%21+8;
    // n = 30;     // set 30 to test

    int i, row, col;

    // Generate the coefficient A
    for (row = 0; row < n; row++) {
        for (col = 0; col < n; col++) {
            A[row][col] = rand()%100+1;
        }
    }

    // Generate the solution x
    printf("Generate the random augmented matrix with solutions: ");
    for (i = 0; i<n; i++) {
        x[i] = rand()%201 - 100; 
        printf("%.1f, ",x[i]);
    }
    printf("\n");
    
    // Generate the linear equations by b=Ax
    for (row = 0; row<n; row++) {
        for (col = 0; col < n; col++) {
            A[row][n] += A[row][col] * x[col];
        }
    }    
}

void gauss() {
// Gaussian Elimination
    int k;
    int i,j;
    int row;

    // Process 0 broadcasts n to other processes.
    MPI_Bcast(&n, 1, MPI_INT,0,MPI_COMM_WORLD);

    for (k = 0; k<n-1 ; k++){
        
        // Process 0 broadcasts A to other processes.
        MPI_Bcast(&A[k][0], n+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (my_rank == 0){

            MPI_Request request[n];

            for (i = 1; i<comm_sz; i++){
                for (row = k+1+i; row < n; row = row + comm_sz){

                    // Equally assign task, process 0 sends the rows they need to compute
                    MPI_Isend(&A[row], n+1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request[row]);
                }
            }   

            // While waiting, process 0 computes its own row
            for (i = k+1; i<n; i = i + comm_sz){
                A[i][k] = A[i][k] / A[k][k];
                for (j = k+1; j<n+1; j++){
                    A[i][j] = A[i][j] - A[k][j] * A[i][k];
                }
            }


            for (i = 1; i<comm_sz; i++){
                for (row = k+1+i; row < n; row = row + comm_sz){
                    // Process 0 wait the non-blocking operation MPI_Isend complete
                    MPI_Wait(&request[row], MPI_STATUS_IGNORE);

                    // Process 0 recieve the results from other
                    MPI_Recv(&A[row], n+1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            } 
        }else{
            
            // Other processes recieve the rows and compute respectively
            for (i = k+1+my_rank; i<n; i=i+comm_sz) {
                MPI_Recv(&A[i], n+1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                A[i][k] = A[i][k] / A[k][k];
                for (j = k+1; j<n+1; j++){
                    A[i][j] = A[i][j] - A[k][j] * A[i][k];
                }

                // Send the result to 0
                MPI_Send(&A[i], n+1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }
        }   
        // Wait other processes complete
        MPI_Barrier(MPI_COMM_WORLD);
    }

}

void backSubstitution(){
// Back substitution, solution store at A[i][n]
    int i, j ;
    for (i = n - 1; i >= 0; i--) {

        for (j = i+1; j < n; j++) {

            A[i][n] = A[i][n] - A[i][j] * A[j][n];

        }

        A[i][n] = A[i][n] / A[i][i];
    }    
}

void printMatrix() {
// Print matrix A
    int row, column;
    for (row=0; row<n; row++)
    {
        for(column=0; column<n+1; column++)
        {
            printf("%.2f     ", A[row][column]);
        }
        printf("\n");
    }
    printf("-------------------\n");
}

void printSolution() {
// Print the solutions
    int row;
    printf("The solution is ");
    for (row = 0; row<n; row++){
        printf("%.2f, ", A[row][n]);
    }
    printf("\n");
}