#include "main.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++17 main.cpp -o main
// RUN COMMAND (4 Processors): mpirun -np 4 ./main

#define original_x_res 1 // Original resolution of the temperature grid, x-axis
#define original_y_res 1 // Original resolution of the temperature grid, y-axis

#define interpolated_x_res 0.1 // Resolution of the interpolated temperature grid, x-axis
#define interpolated_y_res 0.1 // Resolution of the interpolated temperature grid, y-axis

// Driver function to run the interpolation program
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); // Initialize MPI

    int iProc, nProcs; // Rank of the processor and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc); // Get the rank of the processor
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // Get the total number of processors

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors

    // Check if the number of processors is a perfect square
    int sqrt_procs = std::sqrt(nProcs);
    if (iProc == 0) {
        if (sqrt_procs * sqrt_procs != nProcs) {
            MPI_Finalize();
            throw std::runtime_error("Number of processors must be a perfect square");
        }
    }

    // Create the original temperature grid
    int total_spatial_width = 10; // Total spatial width of the temperature grid
    double x_resolution = original_x_res; // Resolution of the temperature grid, x-axis
    double y_resolution = original_y_res; // Resolution of the temperature grid, y-axis
    int num_x_indices = total_spatial_width / x_resolution; // Number of x indices in the temperature grid
    int num_y_indices = total_spatial_width / y_resolution; // Number of y indices in the temperature grid

    int x_indices_per_proc = num_x_indices / sqrt_procs; // Number of x indices per processor
    int y_indices_per_proc = num_y_indices / sqrt_procs; // Number of y indices per processor

    int x_start = (iProc / sqrt_procs) * x_indices_per_proc; // Starting x index for the processor
    int y_start = (iProc % sqrt_procs) * y_indices_per_proc; // Starting y index for the processor

    // Create the original temperature grid for the processor
    SpatialGrid local_temperature_grid(x_indices_per_proc, y_indices_per_proc, total_spatial_width, x_resolution, y_resolution, x_start, y_start);

    // Initialize the temperature grid with a sine wave based on distance from origin
    for (int i = 0; i < x_indices_per_proc; i++) {
        for (int j = 0; j < y_indices_per_proc; j++) {
            auto [x_dist, y_dist] = local_temperature_grid.get_global_coords(i, j);
            double dist = std::sqrt(x_dist * x_dist + y_dist * y_dist);
            double temp = 200 + 100 * std::sin(dist * 25.0 * M_PI / 100);
            local_temperature_grid.set(x_dist, y_dist, temp);
        }
    }

    // Create a boundary map for the temperature grid
    std::map<std::pair<double, double>, double> local_boundary_map;
    local_temperature_grid.create_boundary_map(local_boundary_map);

    // Pack the boundary map into a vector
    std::vector<double> local_boundary_vector;
    local_boundary_vector = local_temperature_grid.pack_boundary_map(local_boundary_map);

    // Gather the boundary vectors from all processors
    std::vector<double> global_boundary_vector(nProcs * local_boundary_vector.size());
    MPI_Allgather(local_boundary_vector.data(), local_boundary_vector.size(), MPI_DOUBLE, global_boundary_vector.data(), local_boundary_vector.size(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Unpack the boundary vectors into a global boundary map
    std::map<std::pair<double, double>, double> global_boundary_map;
    global_boundary_map = local_temperature_grid.unpack_boundary_vector(global_boundary_vector);

    // Interpolate the temperature grid
    std::vector<double> local_temperature_vector;
    for (int i = 0; i < x_indices_per_proc; i++) {
        for (int j = 0; j < y_indices_per_proc; j++) {
            auto [x, y] = local_temperature_grid.get_global_coords(i, j);
            local_temperature_vector.push_back(local_temperature_grid.get(x, y));
        }
    }

    std::vector<double> global_temperature_vector(nProcs * x_indices_per_proc * y_indices_per_proc); // Vector to store the global temperature grid

    // Gather the local temperature vectors from all processors
    MPI_Gather(local_temperature_vector.data(), x_indices_per_proc * y_indices_per_proc, MPI_DOUBLE, global_temperature_vector.data(), x_indices_per_proc * y_indices_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print the global temperature grid
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

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors

    // Create the interpolated temperature grid
    int total_spatial_width_new = 10; // Total spatial width of the interpolated temperature grid
    double x_resolution_new = interpolated_x_res; // Resolution of the interpolated temperature grid, x-axis
    double y_resolution_new = interpolated_y_res; // Resolution of the interpolated temperature grid, y-axis
    int num_x_indices_new = total_spatial_width_new / x_resolution_new; // Number of x indices in the interpolated temperature grid
    int num_y_indices_new = total_spatial_width_new / y_resolution_new; // Number of y indices in the interpolated temperature grid

    int x_indices_per_proc_new = num_x_indices_new / sqrt_procs; // Number of x indices per processor
    int y_indices_per_proc_new = num_y_indices_new / sqrt_procs; //     Number of y indices per processor

    int x_start_new = (iProc / sqrt_procs) * x_indices_per_proc_new; // Starting x index for the processor
    int y_start_new = (iProc % sqrt_procs) * y_indices_per_proc_new; // Starting y index for the processor

    // Create the interpolated temperature grid for the processor
    SpatialGrid local_interpolated_grid(x_indices_per_proc_new, y_indices_per_proc_new, total_spatial_width_new, x_resolution_new, y_resolution_new, x_start_new, y_start_new);

    // Interpolate the temperature grid
    for (int i = 0; i < x_indices_per_proc_new; i++) {
        for (int j = 0; j < y_indices_per_proc_new; j++) {
            auto [x, y] = local_interpolated_grid.get_global_coords(i, j);
            MPI_Barrier(MPI_COMM_WORLD);
            SpatialGrid::transfer_coord(iProc, nProcs, x, y, local_temperature_grid, local_interpolated_grid, global_boundary_map);
        }
    }

    // Gather the interpolated temperature vectors from all processors
    std::vector<double> local_interpolated_vector;
    for (int i = 0; i < x_indices_per_proc_new; i++) {
        for (int j = 0; j < y_indices_per_proc_new; j++) {
            auto [x, y] = local_interpolated_grid.get_global_coords(i, j);
            local_interpolated_vector.push_back(local_interpolated_grid.get(x, y));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors

    // Gather the local interpolated vectors from all processors
    std::vector<double> global_interpolated_vector(nProcs * x_indices_per_proc_new * y_indices_per_proc_new);
    MPI_Gather(local_interpolated_vector.data(), x_indices_per_proc_new * y_indices_per_proc_new, MPI_DOUBLE, global_interpolated_vector.data(), x_indices_per_proc_new * y_indices_per_proc_new, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print the global interpolated grid
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

    MPI_Finalize(); // Finalize MPI
}