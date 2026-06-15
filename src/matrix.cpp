#include "../include/matrix.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace MetalGEMM {

Matrix::Matrix(size_t rows, size_t cols, f32 val)
    : rows_(rows), cols_(cols), data_(rows * cols, val) {}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<f32> &data)
    : rows_(rows), cols_(cols), data_(data) {
    assert(data.size() == rows * cols);
}

Matrix Matrix::random(size_t rows, size_t cols, f32 min, f32 max) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<f32> dist(min, max);
    Matrix m(rows, cols);
    for (auto &v : m.data_) v = dist(rng);
    return m;
}

Matrix Matrix::identity(size_t n) {
    Matrix m(n, n, 0.0f);
    for (size_t i = 0; i < n; i++) m(i, i) = 1.0f;
    return m;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0f);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0f);
}

f32 &Matrix::at(size_t row, size_t col) {
    assert(row < rows_ && col < cols_);
    return data_[row * cols_ + col];
}

const f32 &Matrix::at(size_t row, size_t col) const {
    assert(row < rows_ && col < cols_);
    return data_[row * cols_ + col];
}

f32 *Matrix::data() {
    return data_.data();
}

const f32 *Matrix::data() const {
    return data_.data();
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

f32 Matrix::max_diff(const Matrix &other) const {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    f32 max_d = 0.0f;
    for (size_t i = 0; i < data_.size(); i++) {
        max_d = std::max(max_d, std::abs(data_[i] - other.data_[i]));
    }
    return max_d;
}

f32 Matrix::frobenius_norm() const {
    f32 sum = 0.0f;
    for (auto v : data_) sum += v * v;
    return std::sqrt(sum);
}

bool Matrix::approx_equal(const Matrix &other, f32 tol) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) return false;
    return max_diff(other) <= tol;
}

void Matrix::fill(f32 val) {
    std::fill(data_.begin(), data_.end(), val);
}

void Matrix::print(const std::string &name, size_t max_rows, size_t max_cols) const {
    if (!name.empty()) std::cout << name << " [" << rows_ << "x" << cols_ << "]:\n";
    size_t r = std::min(rows_, max_rows);
    size_t c = std::min(cols_, max_cols);
    for (size_t i = 0; i < r; i++) {
        std::cout << "  ";
        for (size_t j = 0; j < c; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << (*this)(i, j);
        }
        if (c < cols_) std::cout << " ...";
        std::cout << "\n";
    }
    if (r < rows_) std::cout << "  ...\n";
}

} // namespace MetalGEMM
