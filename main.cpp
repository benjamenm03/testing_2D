// main.cpp
#include "main.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

#define ORIGINAL_X_RES 1.0
#define ORIGINAL_Y_RES 1.0
#define INTERP_X_RES   0.1
#define INTERP_Y_RES   0.1

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int iProc, nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    int sqrt_p = static_cast<int>(std::sqrt(nProcs));
    if (sqrt_p * sqrt_p != nProcs) {
        if (iProc == 0) {
            std::cerr << "Number of processes must be a perfect square" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // --- original grid setup on all ranks ---
    const int total_width_orig = 10;
    const double xres_o = ORIGINAL_X_RES;
    const double yres_o = ORIGINAL_Y_RES;
    const int nx_o = static_cast<int>(total_width_orig / xres_o);
    const int ny_o = static_cast<int>(total_width_orig / yres_o);
    const int px_o = nx_o / sqrt_p;
    const int py_o = ny_o / sqrt_p;
    const int xs_o = (iProc / sqrt_p) * px_o;
    const int ys_o = (iProc % sqrt_p) * py_o;

    SpatialGrid local_src(px_o, py_o, total_width_orig, xres_o, yres_o, xs_o, ys_o);
    for (int i = 0; i < px_o; ++i) {
        for (int j = 0; j < py_o; ++j) {
            auto [x, y] = local_src.get_global_coords(i, j);
            double dist = std::sqrt(x*x + y*y);
            double temp = 200 + 100 * std::sin(dist * 25.0 * M_PI / 100);
            local_src.set(x, y, temp);
        }
    }

    // gather full source grid on rank 0
    std::vector<double> local_vals(px_o * py_o);
    for (int idx = 0, i = 0; i < px_o; ++i) {
        for (int j = 0; j < py_o; ++j, ++idx) {
            auto [x, y] = local_src.get_global_coords(i, j);
            local_vals[idx] = local_src.get(x, y);
        }
    }
    std::vector<double> global_vals;
    if (iProc == 0) global_vals.resize(nProcs * px_o * py_o);
    MPI_Gather(local_vals.data(), px_o * py_o, MPI_DOUBLE,
               global_vals.data(), px_o * py_o, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (iProc == 0) {
        // reconstruct full source grid
        SpatialGrid full_src(nx_o, ny_o, total_width_orig, xres_o, yres_o, 0, 0);
        for (int p = 0; p < nProcs; ++p) {
            int bx = (p / sqrt_p) * px_o;
            int by = (p % sqrt_p) * py_o;
            for (int i = 0; i < px_o; ++i) {
                for (int j = 0; j < py_o; ++j) {
                    int idx = p * (px_o * py_o) + i * py_o + j;
                    double val = global_vals[idx];
                    auto [gx, gy] = full_src.get_global_coords(i + bx, j + by);
                    full_src.set(gx, gy, val);
                }
            }
        }

        std::cout << "Original grid:" << std::endl;
        full_src.print_to_terminal();
        full_src.print_to_csv("temperature_grid.csv");

        // --- interpolation on rank 0 only ---
        const int total_width_new = 12;
        const double xres_n = INTERP_X_RES;
        const double yres_n = INTERP_Y_RES;
        const int nx_n = static_cast<int>(total_width_new / xres_n);
        const int ny_n = static_cast<int>(total_width_new / yres_n);

        SpatialGrid interp_grid(nx_n, ny_n, total_width_new, xres_n, yres_n, 0, 0);
        for (int i = 0; i < nx_n; ++i) {
            for (int j = 0; j < ny_n; ++j) {
                double x = i * xres_n;
                double y = j * yres_n;
                double value;
                if (full_src.is_valid_coord(x, y)) {
                    value = full_src.get(x, y);
                } else if (x >= 0 && x <= full_src.x_max_boundary &&
                           y >= 0 && y <= full_src.y_max_boundary) {
                    auto nbs = full_src.find_nearest_coords(x, y, xres_o, yres_o);
                    double x1 = nbs[0].first, y1 = nbs[0].second;
                    double x2 = nbs[1].first, y2 = nbs[1].second;
                    double x3 = nbs[2].first, y3 = nbs[2].second;
                    double x4 = nbs[3].first, y4 = nbs[3].second;
                    double v1 = full_src.get(x1, y1);
                    double v2 = full_src.get(x2, y2);
                    double v3 = full_src.get(x3, y3);
                    double v4 = full_src.get(x4, y4);
                    double t = (x - x1) / (x2 - x1);
                    double u = (y - y1) / (y3 - y1);
                    value = (1 - t) * (1 - u) * v1 + t * (1 - u) * v2
                            + (1 - t) * u * v3 + t * u * v4;
                } else {
                    double xc = std::min(std::max(x, 0.0), full_src.x_max_boundary);
                    double yc = std::min(std::max(y, 0.0), full_src.y_max_boundary);
                    value = full_src.get(xc, yc);
                }
                interp_grid.set(x, y, value);
            }
        }

        std::cout << "Interpolated grid:" << std::endl;
        interp_grid.print_to_terminal();
        interp_grid.print_to_csv("interpolated_grid.csv");
    }

    MPI_Finalize();
    return 0;
}
