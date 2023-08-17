#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>


class Matrix {
private:
    size_t rows;
    size_t columns;
    std::vector<std::vector<double>> _data;

public:
    Matrix(size_t rows, size_t columns);
    explicit Matrix(const std::vector<std::vector<double>>&);
    Matrix(const std::vector<double>&);

    Matrix(const Matrix& other);
    Matrix(Matrix&& other);
    Matrix& operator =(const Matrix& other);
    Matrix& operator =(Matrix&& other);
    ~Matrix() = default;

    const std::vector<double>& operator[](size_t) const;
    std::vector<double>& operator[](size_t);
    size_t get_rows() const;
    size_t get_columns() const;

    static Matrix hadamard_multiply(const Matrix&, const Matrix&);
    static Matrix transpose(const Matrix&);

    Matrix& operator +=(const Matrix&);
    Matrix& operator -=(const Matrix&);
    Matrix& operator *=(double);
    Matrix& operator /=(double);

};

Matrix operator *(const Matrix&, const Matrix&);
Matrix operator *(double, const Matrix&);
Matrix operator *(const Matrix&, double);

Matrix operator /(double, const Matrix&);
Matrix operator /(const Matrix&, double);

Matrix operator -(const Matrix&, const Matrix&);
Matrix operator +(const Matrix&, const Matrix&);

bool operator==(const Matrix& lhs, const Matrix& rhs);
bool operator!=(const Matrix& lhs, const Matrix& rhs);

std::ostream& operator<<(std::ostream& out, const Matrix&);

#endif // MATRIX_H
