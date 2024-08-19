#ifndef MAIN_H
#define MAIN_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <armadillo>
#include <fstream>

class SpatialGrid {
public:
    arma::mat grid;
    double x_resolution, y_resolution;
    int x_start, y_start;
    int x_indices_per_proc, y_indices_per_proc;
    int total_spatial_width;

    SpatialGrid(int x_indices_per_proc, int y_indices_per_proc, int total_spatial_width, double x_resolution, double y_resolution, int x_start = 0, int y_start = 0) {
        this->grid = arma::zeros<arma::mat>(x_indices_per_proc, y_indices_per_proc);
        this->x_resolution = x_resolution;
        this->y_resolution = y_resolution;
        this->x_start = x_start;
        this->y_start = y_start;
        this->x_indices_per_proc = x_indices_per_proc;
        this->y_indices_per_proc = y_indices_per_proc;
        this->total_spatial_width = total_spatial_width;
    }

    void set(double x, double y, double value) {
        const double tolerance = 1e-9;
        int i = static_cast<int>((x - x_start * x_resolution + tolerance) / x_resolution + 0.5);
        int j = static_cast<int>((y - y_start * y_resolution + tolerance) / y_resolution + 0.5);
        if (i >= 0 && i < x_indices_per_proc && j >= 0 && j < y_indices_per_proc) {
            grid(i, j) = value;
        } else {
            std::cout << "Invalid coordinate in set: (" << x << ", " << y << ")" << std::endl;
            throw std::runtime_error("Invalid coordinate in set");
        }
    }

    double get(double x, double y) {
        const double tolerance = 1e-9;
        int i = static_cast<int>((x - x_start * x_resolution + tolerance) / x_resolution + 0.5);
        int j = static_cast<int>((y - y_start * y_resolution + tolerance) / y_resolution + 0.5);
        if (i >= 0 && i < x_indices_per_proc && j >= 0 && j < y_indices_per_proc) {
            return grid(i, j);
        } else {
            std::cout << "Invalid coordinate in get: (" << x << ", " << y << ")" << std::endl;
            throw std::runtime_error("Invalid coordinate in get");
        }
    }

    std::pair<double, double> get_global_coords(int i, int j) {
        if (i >= 0 && i < x_indices_per_proc && j >= 0 && j < y_indices_per_proc) {
            return {(x_start + i) * x_resolution, (y_start + j) * y_resolution};
        } else {
            std::cout << "Invalid indices in get_global_coords: (" << i << ", " << j << ")" << std::endl;
            throw std::runtime_error("Invalid indices in get_global_coords");
        }
    }

    int get_owner_coord(double x, double y, int sqrt_procs, double x_res, double y_res) {
        int num_x_indices = total_spatial_width / x_res;
        int num_y_indices = total_spatial_width / y_res;
        int x_indices_per_proc = num_x_indices / sqrt_procs;
        int y_indices_per_proc = num_y_indices / sqrt_procs;
        int x_start = static_cast<int>(x / x_res) / x_indices_per_proc;
        int y_start = static_cast<int>(y / y_res) / y_indices_per_proc;
        return x_start * sqrt_procs + y_start;
    }

    std::vector<std::pair<double, double>> find_nearest_coords(double x, double y, double src_x_resolution, double src_y_resolution, int src_total_width, int src_x_start, int src_y_start) {
        double x_mod = fmod(x, src_x_resolution);
        double y_mod = fmod(y, src_y_resolution);

        double x1 = x - x_mod;
        double y1 = y - y_mod;

        double x2 = x1 + src_x_resolution;
        double y2 = y1;

        double x3 = x1;
        double y3 = y1 + src_y_resolution;

        double x4 = x1 + src_x_resolution;
        double y4 = y1 + src_y_resolution;

        return {{x1, y1}, {x2, y2}, {x3, y3}, {x4, y4}};
    }

    static void transfer_coord(int iProc, int nProcs, double x, double y, SpatialGrid &src_grid, SpatialGrid &dest_grid, bool print = false) {
        // *********************** THIS WORKS ***********************
        bool src_exists = src_grid.is_valid_coord(x, y);
        bool dest_exists = dest_grid.is_valid_coord(x, y);

        if (src_exists && dest_exists) {
            // Copy point over
            int src_owner = src_grid.get_owner_coord(x, y, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);
            int dest_owner = dest_grid.get_owner_coord(x, y, std::sqrt(nProcs), dest_grid.x_resolution, dest_grid.y_resolution);

            if (src_owner == iProc && dest_owner == iProc) {
                double value = src_grid.get(x, y);
                dest_grid.set(x, y, value);
                if (print) {
                    std::cout << "Processor " << iProc << " owns coord (" << x << ", " << y << ") and copied value" << std::endl;
                }
            } else if (src_owner == iProc && dest_owner != iProc) {
                double value = src_grid.get(x, y);
                MPI_Send(&value, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
                if (print) {
                    std::cout << "Processor " << iProc << " sent coord (" << x << ", " << y << ") to processor " << dest_owner << std::endl;
                }
            } else if (src_owner != iProc && dest_owner == iProc) {
                double value;
                MPI_Recv(&value, 1, MPI_DOUBLE, src_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                dest_grid.set(x, y, value);
                if (print) {
                    std::cout << "Processor " << iProc << " received coord (" << x << ", " << y << ") from processor " << src_owner << std::endl;
                }
            } else {
                if (print) {
                    std::cout << "Processor " << iProc << " does not own the source or destination coord. Skipping..." << std::endl;
                }
            }
        // **********************************************************
        } else if (!src_exists && dest_exists) {
            // Interpolate

            // 1. Find the nearest coordinates in the source grid
            // 2. Determine owners of each corner of the nearest coordinates rectangle
            // 3. Look through combination of ownership possibilities
                // a. If all four corners are owned by the same processor, do nothing other than interpolate and paste value in
                // b. If each corner is owned by separate processor, each processor sends, and destination processor receives
                // c. If there's a mismatch, each processor that owns a point does a send, and the destination does a receive
            // 4. Interpolate the value and paste it into the destination grid

            std::vector<std::pair<double, double> > nearest_coords = src_grid.find_nearest_coords(x, y, src_grid.x_resolution, src_grid.y_resolution, src_grid.total_spatial_width, src_grid.x_start, src_grid.y_start);

            double x1 = nearest_coords[0].first, y1 = nearest_coords[0].second;
            double x2 = nearest_coords[1].first, y2 = nearest_coords[1].second;
            double x3 = nearest_coords[2].first, y3 = nearest_coords[2].second;
            double x4 = nearest_coords[3].first, y4 = nearest_coords[3].second;

            int top_left_owner = src_grid.get_owner_coord(x1, y1, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);
            int bottom_left_owner = src_grid.get_owner_coord(x2, y2, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);
            int top_right_owner = src_grid.get_owner_coord(x3, y3, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);
            int bottom_right_owner = src_grid.get_owner_coord(x4, y4, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);

            int dest_owner = dest_grid.get_owner_coord(x, y, std::sqrt(nProcs), dest_grid.x_resolution, dest_grid.y_resolution);

            if (top_left_owner == bottom_left_owner && bottom_left_owner == top_right_owner && top_right_owner == bottom_right_owner && bottom_right_owner == dest_owner) {
                double top_left_val = src_grid.get(x1, y1);
                double bottom_left_val = src_grid.get(x2, y2);
                double top_right_val = src_grid.get(x3, y3);
                double bottom_right_val = src_grid.get(x4, y4);

                double t = (x - x1) / (x2 - x1);
                double u = (y - y1) / (y3 - y1);
                double interpolated_val = (1 - t) * (1 - u) * top_left_val + t * (1 - u) * bottom_left_val + (1 - t) * u * top_right_val + t * u * bottom_right_val;

                dest_grid.set(x, y, interpolated_val);
                if (print) {
                    std::cout << "Processor " << iProc << " interpolated coord (" << x << ", " << y << ") and pasted value" << std::endl;
                }
            }
            else {
                dest_grid.set(x, y, 100.0);
            }

        } else if (src_exists && !dest_exists) {
            // Throw error
            std::cout << "Destination grid does not contain coordinate (" << x << ", " << y << ")" << std::endl;
            throw std::runtime_error("Destination grid does not contain coordinate");
        } else {
            // Throw error
            std::cout << "Neither source nor destination grid contain coordinate (" << x << ", " << y << ")" << std::endl;
            throw std::runtime_error("Neither source nor destination grid contain coordinate");
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    bool is_valid_coord(double x, double y) {
        const double tolerance = 1e-9;  // Small tolerance to handle precision issues

        // Correctly validate against the entire grid, not just local
        if (x < 0 || x >= total_spatial_width || y < 0 || y >= total_spatial_width) {
            return false;
        }

        // Ensure the coordinates align with the grid resolution
        double x_mod = fmod(x, x_resolution);
        double y_mod = fmod(y, y_resolution);

        return (std::abs(x_mod) < tolerance || std::abs(x_mod - x_resolution) < tolerance) &&
            (std::abs(y_mod) < tolerance || std::abs(y_mod - y_resolution) < tolerance);
    }


    void print_to_terminal() {
        for (int i = 0; i < x_indices_per_proc; i++) {
            for (int j = 0; j < y_indices_per_proc; j++) {
                double data = grid(i, j);
                std::cout << std::setw(10) << data << " ";
            }
            std::cout << std::endl;
        }
    }

    void print_to_csv(const std::string &filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return;
        }

        for (int i = 0; i < x_indices_per_proc; i++) {
            for (int j = 0; j < y_indices_per_proc; j++) {
                double data = grid(i, j);
                file << data;
                if (j < y_indices_per_proc - 1) {
                    file << ",";
                }
            }
            file << std::endl;
        }

        file.close();
    }
};

#endif