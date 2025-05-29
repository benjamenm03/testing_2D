// main.h
#ifndef MAIN_H
#define MAIN_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <armadillo>
#include <fstream>
#include <stdexcept>
#include <algorithm>

class SpatialGrid {
public:
    arma::mat grid;
    double x_resolution, y_resolution;
    int x_start, y_start;
    int x_indices_per_proc, y_indices_per_proc;
    int total_spatial_width;
    double x_max_boundary, y_max_boundary;

    SpatialGrid(int x_indices_per_proc, int y_indices_per_proc, int total_spatial_width,
                double x_resolution, double y_resolution, int x_start = 0, int y_start = 0)
        : grid(arma::zeros<arma::mat>(x_indices_per_proc, y_indices_per_proc)),
          x_resolution(x_resolution), y_resolution(y_resolution),
          x_start(x_start), y_start(y_start),
          x_indices_per_proc(x_indices_per_proc), y_indices_per_proc(y_indices_per_proc),
          total_spatial_width(total_spatial_width),
          x_max_boundary(total_spatial_width - x_resolution),
          y_max_boundary(total_spatial_width - y_resolution) {}

    void set(double x, double y, double value) {
        const double tol = 1e-9;
        int i = static_cast<int>((x - x_start * x_resolution + tol) / x_resolution + 0.5);
        int j = static_cast<int>((y - y_start * y_resolution + tol) / y_resolution + 0.5);
        if (i >= 0 && i < x_indices_per_proc && j >= 0 && j < y_indices_per_proc) {
            grid(i, j) = value;
        }
    }

    double get(double x, double y) const {
        // clamp to local grid bounds for both source and destination
        double min_x = x_start * x_resolution;
        double max_x = (x_start + x_indices_per_proc - 1) * x_resolution;
        double min_y = y_start * y_resolution;
        double max_y = (y_start + y_indices_per_proc - 1) * y_resolution;
        x = std::min(std::max(x, min_x), max_x);
        y = std::min(std::max(y, min_y), max_y);
        const double tol = 1e-9;
        int i = static_cast<int>((x - x_start * x_resolution + tol) / x_resolution + 0.5);
        int j = static_cast<int>((y - y_start * y_resolution + tol) / y_resolution + 0.5);
        return grid(i, j);
    }

    bool is_valid_coord(double x, double y) const {
        const double tol = 1e-9;
        if (x < 0 || x > x_max_boundary || y < 0 || y > y_max_boundary) return false;
        double xm = std::fmod(x, x_resolution);
        double ym = std::fmod(y, y_resolution);
        return (std::abs(xm) < tol || std::abs(xm - x_resolution) < tol) &&
               (std::abs(ym) < tol || std::abs(ym - y_resolution) < tol);
    }

    std::pair<double, double> get_global_coords(int i, int j) const {
        return { (x_start + i) * x_resolution,
                 (y_start + j) * y_resolution };
    }

    std::vector<std::pair<double, double>> find_nearest_coords(double x, double y,
                                                              double src_x_resolution,
                                                              double src_y_resolution) const {
        double xm = std::fmod(x, src_x_resolution);
        double ym = std::fmod(y, src_y_resolution);
        double x1 = x - xm;
        double y1 = y - ym;
        double x2 = x1 + src_x_resolution;
        double y2 = y1;
        double x3 = x1;
        double y3 = y1 + src_y_resolution;
        double x4 = x1 + src_x_resolution;
        double y4 = y1 + src_y_resolution;
        return {{x1, y1}, {x2, y2}, {x3, y3}, {x4, y4}};
    }

    void print_to_terminal() const {
        for (int i = 0; i < x_indices_per_proc; ++i) {
            for (int j = 0; j < y_indices_per_proc; ++j) {
                std::cout << std::setw(10) << grid(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

    void print_to_csv(const std::string& fname) const {
        std::ofstream f(fname);
        if (!f) { std::cerr << "Unable to open " << fname << '\n'; return; }
        for (int i = 0; i < x_indices_per_proc; ++i) {
            for (int j = 0; j < y_indices_per_proc; ++j) {
                f << grid(i, j) << (j + 1 < y_indices_per_proc ? "," : "\n");
            }
        }
    }
};

#endif // MAIN_H