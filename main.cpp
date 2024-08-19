#include "main.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++17 main.cpp -o main
// RUN COMMAND (4 Processors): mpirun -np 4 ./main

int main(int argc, char **argv) {
    // ********** Debug booleans to print testing statements **********
    
    // ****************************************************************

    MPI_Init(&argc, &argv); // Initialize MPI

    int iProc, nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    MPI_Barrier(MPI_COMM_WORLD);

    int sqrt_procs = std::sqrt(nProcs);
    if (iProc == 0) {
        if (sqrt_procs * sqrt_procs != nProcs) {
            MPI_Finalize();
            throw std::runtime_error("Number of processors must be a perfect square");
        }
    }

    int total_spatial_width = 10;
    double x_resolution = 1;
    double y_resolution = 1;
    int num_x_indices = total_spatial_width / x_resolution;
    int num_y_indices = total_spatial_width / y_resolution;

    int x_indices_per_proc = num_x_indices / sqrt_procs;
    int y_indices_per_proc = num_y_indices / sqrt_procs;

    int x_start = (iProc / sqrt_procs) * x_indices_per_proc;
    int y_start = (iProc % sqrt_procs) * y_indices_per_proc;

    SpatialGrid local_temperature_grid(x_indices_per_proc, y_indices_per_proc, total_spatial_width, x_resolution, y_resolution, x_start, y_start);

    for (int i = 0; i < x_indices_per_proc; i++) {
        for (int j = 0; j < y_indices_per_proc; j++) {
            auto [x_dist, y_dist] = local_temperature_grid.get_global_coords(i, j);
            double dist = std::sqrt(x_dist * x_dist + y_dist * y_dist);
            double temp = 200 + 100 * std::sin(dist * 25.0 * M_PI / 100);
            local_temperature_grid.set(x_dist, y_dist, temp);
        }
    }

    std::vector<double> local_temperature_vector;
    for (int i = 0; i < x_indices_per_proc; i++) {
        for (int j = 0; j < y_indices_per_proc; j++) {
            auto [x, y] = local_temperature_grid.get_global_coords(i, j);
            local_temperature_vector.push_back(local_temperature_grid.get(x, y));
        }
    }

    std::vector<double> global_temperature_vector(nProcs * x_indices_per_proc * y_indices_per_proc);

    MPI_Gather(local_temperature_vector.data(), x_indices_per_proc * y_indices_per_proc, MPI_DOUBLE, global_temperature_vector.data(), x_indices_per_proc * y_indices_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (iProc == 0) {
        SpatialGrid global_temperature_grid(num_x_indices, num_y_indices, total_spatial_width, x_resolution, y_resolution);

        for (int proc = 0; proc < nProcs; proc++) {
            int proc_x_start = (proc / sqrt_procs) * x_indices_per_proc;
            int proc_y_start = (proc % sqrt_procs) * y_indices_per_proc;

            for (int i = 0; i < x_indices_per_proc; i++) {
                for (int j = 0; j < y_indices_per_proc; j++) {
                    double value = global_temperature_vector[proc * x_indices_per_proc * y_indices_per_proc + i * y_indices_per_proc + j];
                    global_temperature_grid.set((proc_x_start + i) * x_resolution, (proc_y_start + j) * y_resolution, value);
                }
            }
        }

        global_temperature_grid.print_to_terminal();
        global_temperature_grid.print_to_csv("temperature_grid.csv");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int total_spatial_width_new = 10;
    double x_resolution_new = 0.5;
    double y_resolution_new = 0.5;
    int num_x_indices_new = total_spatial_width_new / x_resolution_new;
    int num_y_indices_new = total_spatial_width_new / y_resolution_new;

    int x_indices_per_proc_new = num_x_indices_new / sqrt_procs;
    int y_indices_per_proc_new = num_y_indices_new / sqrt_procs;

    int x_start_new = (iProc / sqrt_procs) * x_indices_per_proc_new;
    int y_start_new = (iProc % sqrt_procs) * y_indices_per_proc_new;

    SpatialGrid local_interpolated_grid(x_indices_per_proc_new, y_indices_per_proc_new, total_spatial_width_new, x_resolution_new, y_resolution_new, x_start_new, y_start_new);

    for (int i = 0; i < x_indices_per_proc_new; i++) {
        for (int j = 0; j < y_indices_per_proc_new; j++) {
            auto [x, y] = local_interpolated_grid.get_global_coords(i, j);
            MPI_Barrier(MPI_COMM_WORLD);
            SpatialGrid::transfer_coord(iProc, nProcs, x, y, local_temperature_grid, local_interpolated_grid);
        }
    }

    std::vector<double> local_interpolated_vector;
    for (int i = 0; i < x_indices_per_proc_new; i++) {
        for (int j = 0; j < y_indices_per_proc_new; j++) {
            auto [x, y] = local_interpolated_grid.get_global_coords(i, j);
            local_interpolated_vector.push_back(local_interpolated_grid.get(x, y));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> global_interpolated_vector(nProcs * x_indices_per_proc_new * y_indices_per_proc_new);
    MPI_Gather(local_interpolated_vector.data(), x_indices_per_proc_new * y_indices_per_proc_new, MPI_DOUBLE, global_interpolated_vector.data(), x_indices_per_proc_new * y_indices_per_proc_new, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (iProc == 0) {
        SpatialGrid global_interpolated_grid(num_x_indices_new, num_y_indices_new, total_spatial_width_new, x_resolution_new, y_resolution_new);

        for (int proc = 0; proc < nProcs; proc++) {
            int proc_x_start_new = (proc / sqrt_procs) * x_indices_per_proc_new;
            int proc_y_start_new = (proc % sqrt_procs) * y_indices_per_proc_new;

            for (int i = 0; i < x_indices_per_proc_new; i++) {
                for (int j = 0; j < y_indices_per_proc_new; j++) {
                    double value = global_interpolated_vector[proc * x_indices_per_proc_new * y_indices_per_proc_new + i * y_indices_per_proc_new + j];
                    global_interpolated_grid.set((proc_x_start_new + i) * x_resolution_new, (proc_y_start_new + j) * y_resolution_new, value);
                }
            }
        }

        global_interpolated_grid.print_to_terminal();
        global_interpolated_grid.print_to_csv("interpolated_grid.csv");
    }

    MPI_Finalize();
}