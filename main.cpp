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

    const double total_width_orig_x = 10.0;
    const double total_width_orig_y = 10.0;
    const double xres_o = ORIGINAL_X_RES;
    const double yres_o = ORIGINAL_Y_RES;

    auto compute_grid_points = [](double span, double res) -> int {
        if (res <= 0.0) return 0;
        int pts = static_cast<int>(std::lround(span / res));
        return std::max(1, pts);
    };

    auto compute_counts_offsets = [](int global_n, int p, std::vector<int>& counts,
                                     std::vector<int>& offsets) {
        counts.resize(p);
        offsets.resize(p);
        int base = global_n / p;
        int rem = global_n % p;
        int offset = 0;
        for (int i = 0; i < p; ++i) {
            counts[i] = base + (i < rem ? 1 : 0);
            offsets[i] = offset;
            offset += counts[i];
        }
    };

    int nx_o = compute_grid_points(total_width_orig_x, xres_o);
    int ny_o = compute_grid_points(total_width_orig_y, yres_o);
    if (nx_o == 0 || ny_o == 0) {
        if (iProc == 0) {
            std::cerr << "Invalid grid resolution for original grid" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    double normalized_width_orig_x = nx_o * xres_o;
    double normalized_width_orig_y = ny_o * yres_o;

    std::vector<int> x_counts, x_offsets, y_counts, y_offsets;
    compute_counts_offsets(nx_o, sqrt_p, x_counts, x_offsets);
    compute_counts_offsets(ny_o, sqrt_p, y_counts, y_offsets);

    int row = iProc / sqrt_p;
    int col = iProc % sqrt_p;

    int local_nx = x_counts[row];
    int local_ny = y_counts[col];
    int x_start_idx = x_offsets[row];
    int y_start_idx = y_offsets[col];

    SpatialGrid local_src(local_nx, local_ny,
                          normalized_width_orig_x, normalized_width_orig_y,
                          xres_o, yres_o,
                          x_start_idx, y_start_idx);

    for (int i = 0; i < local_nx; ++i) {
        for (int j = 0; j < local_ny; ++j) {
            auto [x, y] = local_src.get_global_coords(i, j);
            double dist = std::sqrt(x * x + y * y);
            double temp = 200 + 100 * std::sin(dist * 25.0 * M_PI / 100);
            local_src.set(x, y, temp);
        }
    }

    int local_count = local_nx * local_ny;
    std::vector<double> local_vals(local_count);
    for (int idx = 0, i = 0; i < local_nx; ++i) {
        for (int j = 0; j < local_ny; ++j, ++idx) {
            auto [x, y] = local_src.get_global_coords(i, j);
            local_vals[idx] = local_src.get(x, y);
        }
    }

    std::vector<int> sendcounts(nProcs), displs(nProcs);
    std::vector<int> proc_x_counts(nProcs), proc_y_counts(nProcs);
    std::vector<int> proc_x_offsets(nProcs), proc_y_offsets(nProcs);
    int accum = 0;
    for (int p = 0; p < nProcs; ++p) {
        int prow = p / sqrt_p;
        int pcol = p % sqrt_p;
        proc_x_counts[p] = x_counts[prow];
        proc_y_counts[p] = y_counts[pcol];
        proc_x_offsets[p] = x_offsets[prow];
        proc_y_offsets[p] = y_offsets[pcol];
        sendcounts[p] = proc_x_counts[p] * proc_y_counts[p];
        displs[p] = accum;
        accum += sendcounts[p];
    }

    int expected_points = nx_o * ny_o;
    if (expected_points != accum) {
        if (iProc == 0) {
            std::cerr << "Partition mismatch for original grid" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::vector<double> global_vals(expected_points);
    MPI_Allgatherv(local_vals.data(), local_count, MPI_DOUBLE,
                   global_vals.data(), sendcounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    SpatialGrid full_src(nx_o, ny_o,
                         normalized_width_orig_x, normalized_width_orig_y,
                         xres_o, yres_o, 0, 0);
    for (int p = 0; p < nProcs; ++p) {
        int base = displs[p];
        int lx = proc_x_counts[p];
        int ly = proc_y_counts[p];
        int ox = proc_x_offsets[p];
        int oy = proc_y_offsets[p];
        for (int i = 0; i < lx; ++i) {
            for (int j = 0; j < ly; ++j) {
                double x = (ox + i) * xres_o;
                double y = (oy + j) * yres_o;
                full_src.set(x, y, global_vals[base + i * ly + j]);
            }
        }
    }

    if (iProc == 0) {
        std::cout << "Original grid:" << std::endl;
        full_src.print_to_terminal();
        full_src.print_to_csv("temperature_grid.csv");
    }

    const double total_width_new_x = 12.0;
    const double total_width_new_y = 12.0;
    const double xres_n = INTERP_X_RES;
    const double yres_n = INTERP_Y_RES;

    int nx_n = compute_grid_points(total_width_new_x, xres_n);
    int ny_n = compute_grid_points(total_width_new_y, yres_n);
    if (nx_n == 0 || ny_n == 0) {
        if (iProc == 0) {
            std::cerr << "Invalid grid resolution for interpolated grid" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    double normalized_width_new_x = nx_n * xres_n;
    double normalized_width_new_y = ny_n * yres_n;

    std::vector<int> new_x_counts, new_x_offsets, new_y_counts, new_y_offsets;
    compute_counts_offsets(nx_n, sqrt_p, new_x_counts, new_x_offsets);
    compute_counts_offsets(ny_n, sqrt_p, new_y_counts, new_y_offsets);

    int local_nx_n = new_x_counts[row];
    int local_ny_n = new_y_counts[col];
    int new_x_start = new_x_offsets[row];
    int new_y_start = new_y_offsets[col];

    SpatialGrid local_interp(local_nx_n, local_ny_n,
                             normalized_width_new_x, normalized_width_new_y,
                             xres_n, yres_n,
                             new_x_start, new_y_start);

    for (int i = 0; i < local_nx_n; ++i) {
        for (int j = 0; j < local_ny_n; ++j) {
            auto [x, y] = local_interp.get_global_coords(i, j);
            double value;
            if (full_src.is_valid_coord(x, y)) {
                value = full_src.get(x, y);
            } else if (x >= 0.0 && x <= full_src.x_max_boundary &&
                       y >= 0.0 && y <= full_src.y_max_boundary) {
                auto nbs = full_src.find_nearest_coords(x, y, xres_o, yres_o);
                double x1 = nbs[0].first, y1 = nbs[0].second;
                double x2 = nbs[1].first, y2 = nbs[1].second;
                double x3 = nbs[2].first, y3 = nbs[2].second;
                double x4 = nbs[3].first, y4 = nbs[3].second;
                double v1 = full_src.get(x1, y1);
                double v2 = full_src.get(x2, y2);
                double v3 = full_src.get(x3, y3);
                double v4 = full_src.get(x4, y4);
                double denom_x = (x2 - x1);
                double denom_y = (y3 - y1);
                double t = denom_x != 0.0 ? (x - x1) / denom_x : 0.0;
                double u = denom_y != 0.0 ? (y - y1) / denom_y : 0.0;
                t = std::min(std::max(t, 0.0), 1.0);
                u = std::min(std::max(u, 0.0), 1.0);
                value = (1 - t) * (1 - u) * v1 + t * (1 - u) * v2
                        + (1 - t) * u * v3 + t * u * v4;
            } else {
                double xc = std::min(std::max(x, 0.0), full_src.x_max_boundary);
                double yc = std::min(std::max(y, 0.0), full_src.y_max_boundary);
                value = full_src.get(xc, yc);
            }
            local_interp.set(x, y, value);
        }
    }

    int local_interp_count = local_nx_n * local_ny_n;
    std::vector<double> interp_local(local_interp_count);
    for (int idx = 0, i = 0; i < local_nx_n; ++i) {
        for (int j = 0; j < local_ny_n; ++j, ++idx) {
            auto [x, y] = local_interp.get_global_coords(i, j);
            interp_local[idx] = local_interp.get(x, y);
        }
    }

    std::vector<int> interp_sendcounts(nProcs), interp_displs(nProcs);
    std::vector<int> interp_x_counts(nProcs), interp_y_counts(nProcs);
    std::vector<int> interp_x_offsets(nProcs), interp_y_offsets(nProcs);
    int interp_accum = 0;
    for (int p = 0; p < nProcs; ++p) {
        int prow = p / sqrt_p;
        int pcol = p % sqrt_p;
        interp_x_counts[p] = new_x_counts[prow];
        interp_y_counts[p] = new_y_counts[pcol];
        interp_x_offsets[p] = new_x_offsets[prow];
        interp_y_offsets[p] = new_y_offsets[pcol];
        interp_sendcounts[p] = interp_x_counts[p] * interp_y_counts[p];
        interp_displs[p] = interp_accum;
        interp_accum += interp_sendcounts[p];
    }

    int expected_interp_points = nx_n * ny_n;
    if (expected_interp_points != interp_accum) {
        if (iProc == 0) {
            std::cerr << "Partition mismatch for interpolated grid" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::vector<double> interp_global;
    if (iProc == 0) interp_global.resize(expected_interp_points);
    MPI_Gatherv(interp_local.data(), local_interp_count, MPI_DOUBLE,
                iProc == 0 ? interp_global.data() : nullptr,
                interp_sendcounts.data(), interp_displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (iProc == 0) {
        SpatialGrid interp_grid(nx_n, ny_n,
                                normalized_width_new_x, normalized_width_new_y,
                                xres_n, yres_n, 0, 0);
        for (int p = 0; p < nProcs; ++p) {
            int base = interp_displs[p];
            int count_x = interp_x_counts[p];
            int count_y = interp_y_counts[p];
            int offset_x = interp_x_offsets[p];
            int offset_y = interp_y_offsets[p];
            for (int i = 0; i < count_x; ++i) {
                for (int j = 0; j < count_y; ++j) {
                    double x = (offset_x + i) * xres_n;
                    double y = (offset_y + j) * yres_n;
                    interp_grid.set(x, y, interp_global[base + i * count_y + j]);
                }
            }
        }

        std::cout << "Interpolated grid:" << std::endl;
        interp_grid.print_to_terminal();
        interp_grid.print_to_csv("interpolated_grid.csv");
    }

    MPI_Finalize();
    return 0;
}
