#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Initialize the row (number of the variables in linear equations)
int n;

// Initialize augmented matrix with big capacity
double A[100][100];
// Initialize temp solution list for the random generator
double x[100];

// Function header
void initializeRandomMatrix();
void gauss();
void backSubstitution();
void printMatrix();
void printSolution();

int main(int argc, char **argv) {

    // Set the wall clock
    struct timeval wallStart, wallEnd;
    gettimeofday(&wallStart, NULL); 

    // Generate matrix
    initializeRandomMatrix();
    printf("Initial argumented matrix: \n");
    printMatrix();
    
    // Gaussian Elimination
    gauss();
    backSubstitution();

    printMatrix();
    printSolution();

    gettimeofday(&wallEnd, NULL); // end time
    double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);
    double WALLtimeTaken = 1.0E-06 * ((wallSecs*1000000) + (wallEnd.tv_usec - wallStart.tv_usec)); // microseconds
    printf("Wall clock run Time: %f  seconds  \n", WALLtimeTaken);
    printf("\n");
    return 0;
}

void initializeRandomMatrix() {
// Generate random matrix by the random seed

    // Since the total time is small, I use the decimal part of the double as random seed
    double start_time = omp_get_wtime();
    printf("%lf\n", start_time);
    int seed = (int)((start_time - (int)start_time) * 1000000);
    printf("%d\n", seed);
    srand(seed);

    n = rand()%21+8;

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

    // Forward Elimination, form the upper Triangular matrix
    for (k = 0; k<n-1 ; k++){
    #pragma omp parallel for shared(A,k,n) private(i,j) 
        for (i = k+1; i<n; i++){

            // Store the pivot into the lower Triangular matrix to save the iteration steps
            A[i][k] = A[i][k] / A[k][k];

            for (j = k+1; j<n+1; j++){
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
        }   
    }
}

void backSubstitution() {
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