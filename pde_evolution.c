#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

void initialize();
void run_simulation();
void create_output_file(int t);
void trade_ghost_cells(double *r_recv, double *l_recv, double *u_recv, double *d_recv);
double update(int i, int j, double prev_Ci, double next_Ci, double prev_Cj, double next_Cj);
bool checker_board();
int l_mype (int mype);
int r_mype (int mype);
int u_mype (int mype);
int d_mype (int mype); 
double find_prev_Ci(int i, int j);
double find_next_Ci(int i, int j);
double find_prev_Cj(int i, int j);
double find_next_Cj(int i, int j);

double *r_cells, *l_cells, *u_cells, *d_cells;
double *r_send, *l_send, *u_send, *d_send;
int TH, N;
bool output;
int Nprocs, mype;
MPI_Status stat;
double const L = 1.0, T = 1.0;
double del_x, del_t, NT;
int h_nprocs, v_nprocs, r, q;
int i_strt, i_end, j_strt, j_end;
double start_time, end_time;
int i, j, p, row, col;
double *C, *Cnew, *Cfinal;

int main(int argc, char* argv[]) {
    // Get command line arguments
    //    Usage: $ mpirun -n <num of MPI ranks> ./adv_final.exe <num of OpenMP threads per node> <output = true(1)|false(0)>
    TH = atoi(argv[1]);
    output = atoi(argv[2]);

    // Initializing MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);

    // Initializing required parameters 
    N = 500;
    del_x = L / (N - 1);
    del_t = 0.5 / N;
    NT = T / del_t;
    h_nprocs = sqrt((double) Nprocs);
    v_nprocs = Nprocs / h_nprocs;
    r = mype % v_nprocs;
    q = mype / v_nprocs;

    // Calculate the start and end indices for the current process
    i_strt = r * N/v_nprocs;
    i_end = (r == v_nprocs - 1) ? (N) : (i_strt + N/v_nprocs);
    j_strt = q * N/v_nprocs;
    j_end = (q == v_nprocs - 1) ? (N) : (j_strt + N/h_nprocs);
    // Calculate the number of rows and columns for the current process
    col = i_end - i_strt;
    row = j_end - j_strt;

    // Allocate memory for the current and new state of the system
    C = malloc(row * col * sizeof(double));
    Cnew = malloc(row * col * sizeof(double));

    // Initialize the system
    initialize();

    // Running the simulation
    MPI_Barrier(MPI_COMM_WORLD);
    if(mype == 0) { start_time = omp_get_wtime(); }
    run_simulation();
    MPI_Barrier(MPI_COMM_WORLD);
    if(mype == 0) { end_time = omp_get_wtime(); }

    // If the current process is the master, print some statistics
    if (mype == 0) {
        printf("No of nodes: %d, threads per node: %d\n", Nprocs, TH);
        printf("Time taken: %f sec\n", end_time - start_time);
        printf("Grind rate: %f cells/sec\n", (N*N*(NT))/(end_time - start_time));
        printf("------------------------------------------\n\n");
    }

    // Close MPI environment and clean up
    MPI_Finalize();
    free(C); free(Cnew); free(Cfinal);
    free(r_cells); free(l_cells); free(u_cells); free(d_cells);
    free(r_send); free(l_send); free(u_send); free(d_send);
    return 0;
}

// Initialize the system with some initial state
void initialize() {
    double x, y;
    for(i = 0; i < col; i++) {
        x = - L/2 + del_x * (i_strt + i);
        for(j = 0; j < row; j++) {
            y = - L/2 + del_x * (j_strt + j);
            if (y >= -0.1 && y <= 0.1) {
                C[i*row + j] = 1.0;
            }
            else {
                C[i*row + j] = 0.0;
            }
        }
    }
}

// Run the simulation for a certain number of time steps
void run_simulation() {
    // Allocate memory for the ghost cells and the cells to be sent to other processes
    r_cells = malloc(col * sizeof(double));
    l_cells = malloc(col * sizeof(double));
    u_cells = malloc(row * sizeof(double)); 
    d_cells = malloc(row * sizeof(double));
    r_send = malloc(col * sizeof(double));
    l_send = malloc(col * sizeof(double));
    u_send = malloc(row * sizeof(double));
    d_send = malloc(row * sizeof(double));

    for(int t=0; t<NT; t++) {
        MPI_Barrier(MPI_COMM_WORLD);

        // If output is enabled and it's the right time step, create an output file
        if(output == 1 && (t == 0 || t == (int) NT/2 || t == (int) NT-1)) 
            create_output_file(t);

        // Exchange ghost cells with other processes
        trade_ghost_cells(r_cells, l_cells, u_cells, d_cells);
        MPI_Barrier(MPI_COMM_WORLD);

        // Update the state of the system in parallel
        #pragma omp parallel for default(none) shared(mype, C, Cnew, N, del_x, del_t, row, col, u_cells, d_cells, r_cells, l_cells, Nprocs) private(i, j) num_threads(TH)
        for(int i=0; i<col; i++) {
            for(int j=0; j<row; j++) {
                Cnew[i * row + j] = update(i, j, find_prev_Ci(i, j), find_next_Ci(i, j), find_prev_Cj(i, j), find_next_Cj(i, j));
            }
        }

        // Swap the current and new state of the system
        double* temp = C;
        C = Cnew;
        Cnew = temp;
    }
}

void create_output_file(int t) {

    // If the current process is the master and there are multiple processes
    if (mype == 0 && Nprocs != 1) {
        // Allocate memory for the final state of the system
        Cfinal = malloc(N * N * sizeof(double));

        // Copy the state of the current process to the final state
        for (i = 0; i < col; i++) {
            for (j = 0; j < row; j++) {
                Cfinal[i*N + j] = C[i*row + j];
            }
        }

        // Receive the state from all other processes and add it to the final state
        for(p = 1; p < Nprocs; p++) {
            int i_st, i_e, j_st, j_e;

            // Receive the start and end indices from the other process
            MPI_Recv(&i_st, 1, MPI_INT, p, 99+p, MPI_COMM_WORLD, &stat);
            MPI_Recv(&i_e, 1, MPI_INT, p, 99+p, MPI_COMM_WORLD, &stat);
            MPI_Recv(&j_st, 1, MPI_INT, p, 99+p, MPI_COMM_WORLD, &stat);
            MPI_Recv(&j_e, 1, MPI_INT, p, 99+p, MPI_COMM_WORLD, &stat);

            // Receive the state from the other process
            double *Clcl = malloc((i_e-i_st)*(j_e-j_st)*sizeof(double));
            MPI_Recv(Clcl, (i_e-i_st)*(j_e-j_st), MPI_DOUBLE, p, 99+p, MPI_COMM_WORLD, &stat);

            // Add the received state to the final state
            int a = 0;
            for (i = 0; i < i_e - i_st; i++) {
                for (j = 0; j < j_e - j_st; j++) {
                    Cfinal[(i+i_st)*N + (j+j_st)] = Clcl[a];
                    a++;
                }
            }

            // Free the received state
            free(Clcl);
        }
    }
    // If the current process is not the master and there are multiple processes
    else if (mype != 0 && Nprocs != 1) {
        // Send the start and end indices to the master process
        MPI_Send(&i_strt, 1, MPI_INT, 0, 99+mype, MPI_COMM_WORLD);
        MPI_Send(&i_end, 1, MPI_INT, 0, 99+mype, MPI_COMM_WORLD);
        MPI_Send(&j_strt, 1, MPI_INT, 0, 99+mype, MPI_COMM_WORLD);
        MPI_Send(&j_end, 1, MPI_INT, 0, 99+mype, MPI_COMM_WORLD);

        // Send the state to the master process
        MPI_Send(C, row * col, MPI_DOUBLE, 0, 99+mype, MPI_COMM_WORLD);
    }
    // If there is only one process
    else {
        // Allocate memory for the final state of the system
        Cfinal = malloc(N * N * sizeof(double));

        // Copy the state of the current process to the final state
        for (i = 0; i < col; i++) {
            for (j = 0; j < row; j++) {
                Cfinal[i*N + j] = C[i*row + j];
            }
        }
    }

    // If the current process is the master, create the output file
    if (mype == 0) {
        char status[5];
        if (t == 0) strcpy(status, "init");
        if (t == (int) NT/2) strcpy(status, "mid");
        if (t == (int) NT-1) strcpy(status, "end");
        char filename[20];

        sprintf(filename, "%s_%d_%d.txt", status, Nprocs, TH);
        FILE* outputFile = fopen(filename, "w");
        for (i = 0; i < N; ++i) {
            for (j = 0; j < N; ++j) {
                fprintf(outputFile, "%f ", Cfinal[i*N + j]);
            }
            fprintf(outputFile, "\n");
        }
        fclose(outputFile);
    }
}

// Exchange ghost cells with other processes
void trade_ghost_cells(double *r_recv, double *l_recv, double *u_recv, double *d_recv) {
    // Packing ghost cells into 4 arrays ready to send
    for (p = 0; p < col; p++) {
        r_send[p] = C[p*row + row - 1];
        l_send[p] = C[p*row];
    }
    for (p = 0; p < row; p++) {
        u_send[p] = C[p];
        d_send[p] = C[row*(col-1) + p];
    }

    if (Nprocs != 1) {
    // Exchanging of ghost cells
        if (checker_board()) {
            if (i_strt != 0)
                MPI_Send(u_send, row, MPI_DOUBLE, u_mype(mype), 100 + mype, MPI_COMM_WORLD);
            if (i_end != N)
                MPI_Send(d_send, row, MPI_DOUBLE, d_mype(mype), 100 + mype, MPI_COMM_WORLD);
            if (j_strt != 0)
                MPI_Send(l_send, col, MPI_DOUBLE, l_mype(mype), 100 + mype, MPI_COMM_WORLD);
            if (j_end != N)
                MPI_Send(r_send, col, MPI_DOUBLE, r_mype(mype), 100 + mype, MPI_COMM_WORLD);
        }
        else {
            if (i_strt != 0)
                MPI_Recv(u_recv, row, MPI_DOUBLE, u_mype(mype), 100 + u_mype(mype), MPI_COMM_WORLD, &stat);
            if (i_end != N)
                MPI_Recv(d_recv, row, MPI_DOUBLE, d_mype(mype), 100 + d_mype(mype), MPI_COMM_WORLD, &stat);
            if (j_strt !=0)
                MPI_Recv(l_recv, col, MPI_DOUBLE, l_mype(mype), 100 + l_mype(mype), MPI_COMM_WORLD, &stat);
            if (j_end != N)
                MPI_Recv(r_recv, col, MPI_DOUBLE, r_mype(mype), 100 + r_mype(mype), MPI_COMM_WORLD, &stat);
        }
        if (checker_board()) {
            if (i_strt != 0)
                MPI_Recv(u_recv, row, MPI_DOUBLE, u_mype(mype), 100 + u_mype(mype), MPI_COMM_WORLD, &stat);
            if (i_end != N)
                MPI_Recv(d_recv, row, MPI_DOUBLE, d_mype(mype), 100 + d_mype(mype), MPI_COMM_WORLD, &stat);
            if (j_strt !=0)
                MPI_Recv(l_recv, col, MPI_DOUBLE, l_mype(mype), 100 + l_mype(mype), MPI_COMM_WORLD, &stat);
            if (j_end != N)
                MPI_Recv(r_recv, col, MPI_DOUBLE, r_mype(mype), 100 + r_mype(mype), MPI_COMM_WORLD, &stat);
        }
        else{
            if (i_strt != 0)
                MPI_Send(u_send, row, MPI_DOUBLE, u_mype(mype), 100 + mype, MPI_COMM_WORLD);
            if (i_end != N)
                MPI_Send(d_send, row, MPI_DOUBLE, d_mype(mype), 100 + mype, MPI_COMM_WORLD);
            if (j_strt !=0)
                MPI_Send(l_send, col, MPI_DOUBLE, l_mype(mype), 100 + mype, MPI_COMM_WORLD);
            if (j_end != N)
                MPI_Send(r_send, col, MPI_DOUBLE, r_mype(mype), 100 + mype, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (i_strt == 0) {
            MPI_Send(u_send, row, MPI_DOUBLE, u_mype(mype), 100 + mype, MPI_COMM_WORLD);
            MPI_Recv(u_recv, row, MPI_DOUBLE, u_mype(mype), 100 + u_mype(mype), MPI_COMM_WORLD, &stat);
        }
        else if (i_end == N) {
            MPI_Recv(d_recv, row, MPI_DOUBLE, d_mype(mype), 100 + d_mype(mype), MPI_COMM_WORLD, &stat);
            MPI_Send(d_send, row, MPI_DOUBLE, d_mype(mype), 100 + mype, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (j_strt == 0) {
            MPI_Send(l_send, col, MPI_DOUBLE, l_mype(mype), 100 + mype, MPI_COMM_WORLD);
            MPI_Recv(l_recv, col, MPI_DOUBLE, l_mype(mype), 100 + l_mype(mype), MPI_COMM_WORLD, &stat);
        }
        else if (j_end == N) {
            MPI_Recv(r_recv, col, MPI_DOUBLE, r_mype(mype), 100 + r_mype(mype), MPI_COMM_WORLD, &stat);
            MPI_Send(r_send, col, MPI_DOUBLE, r_mype(mype), 100 + mype, MPI_COMM_WORLD);
        }
    }
    else {
        r_recv = l_send;
        l_recv = r_send;
        u_recv = d_send;
        d_recv = u_send;
    }
}

// Check if the current process is on the checker board
bool checker_board() {
    return (((mype % (2 * h_nprocs) < h_nprocs) && ((mype % (2 * h_nprocs) % 2) == 0)) || 
            ((mype % (2 * h_nprocs) >= h_nprocs) && ((((mype % (2 * h_nprocs)) - h_nprocs) % 2) == 1)));
}

// Update the state of the system at a certain point
double update(int i, int j, double prev_Ci, double next_Ci, double prev_Cj, double next_Cj) {
    double u = sqrt(2.0) * (- L/2.0 + del_x * (j_strt + j));
    double v = - sqrt(2.0) * (- L/2.0 + del_x * (i_strt + i));
    return (prev_Ci + next_Ci + prev_Cj + next_Cj)/4
    - (del_t/(2*del_x)) * (u*(next_Ci - prev_Ci) + v*(next_Cj - prev_Cj));
}

// Find the left, right, up, and down neighbors of the current process
int l_mype (int mype) { return (mype / h_nprocs == 0) ? (mype + (v_nprocs - 1)*h_nprocs) : (mype - h_nprocs); }
int r_mype (int mype) { return (mype / h_nprocs == v_nprocs - 1) ? (mype - (v_nprocs - 1)*h_nprocs) : (mype + h_nprocs); }
int u_mype (int mype) { return (mype % v_nprocs == 0) ? (mype + v_nprocs - 1) : (mype - 1); }
int d_mype (int mype) { return (mype % v_nprocs == v_nprocs - 1) ? (mype - v_nprocs + 1) : (mype + 1); }

// Find the left, right, up, and down neighbors of the current cell
double find_next_Ci(int i, int j) { return (i == col-1) ? (d_cells[j]) : (C[(i+1)*row + j]); }
double find_prev_Ci(int i, int j) { return (i == 0) ? (u_cells[j]) : (C[(i-1)*row + j]); }
double find_next_Cj(int i, int j) { return (j == row-1) ? (r_cells[i]) : (C[i*row + j + 1]); }
double find_prev_Cj(int i, int j) { return (j == 0) ? (l_cells[i]) : (C[i*row + j -1]); }
