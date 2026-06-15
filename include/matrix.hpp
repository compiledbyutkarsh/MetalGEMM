#pragma once

#include "types.hpp"
#include <vector>
#include <string>
#include <cassert>
#include <random>
#include <iostream>
#include <iomanip>

namespace MetalGEMM {

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_t rows, size_t cols, f32 val = 0.0f);
    Matrix(size_t rows, size_t cols, const std::vector<f32> &data);

    static Matrix random(size_t rows, size_t cols, f32 min = -1.0f, f32 max = 1.0f);
    static Matrix identity(size_t n);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);

    f32        &at(size_t row, size_t col);
    const f32  &at(size_t row, size_t col) const;
    f32        *data();
    const f32  *data() const;

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    size_t bytes() const { return rows_ * cols_ * sizeof(f32); }

    Matrix transpose() const;
    f32    max_diff(const Matrix &other) const;
    f32    frobenius_norm() const;
    bool   approx_equal(const Matrix &other, f32 tol = 1e-3f) const;

    void   fill(f32 val);
    void   print(const std::string &name = "", size_t max_rows = 8, size_t max_cols = 8) const;

    f32       &operator()(size_t r, size_t c)       { return data_[r * cols_ + c]; }
    const f32 &operator()(size_t r, size_t c) const { return data_[r * cols_ + c]; }

private:
    size_t          rows_;
    size_t          cols_;
    std::vector<f32> data_;
};

} // namespace MetalGEMM
