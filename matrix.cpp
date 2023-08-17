#include "matrix.h"



Matrix::Matrix(size_t rows, size_t columns)
    : rows(rows), columns(columns), _data(rows, std::vector<double>(columns)) { }

Matrix::Matrix(const std::vector<double>& data):
    rows(1), columns(data.size()), _data({data}) { }

Matrix::Matrix(const std::vector<std::vector<double>>& data) {
    if (data.size() == 0)
        return;

    columns = data[0].size();
    for(const auto& vec: data){
        if (vec.size() != columns)
            throw std::invalid_argument("it is not matrix!");
    }

    rows = data.size();
    _data = data;
}



Matrix::Matrix(const Matrix& other){
    rows = other.get_rows();
    columns = other.get_columns();
    _data = other._data;
}

Matrix::Matrix(Matrix&& other){
    rows = other.get_rows();
    columns = other.get_columns();
    std::swap(_data, other._data);
}

Matrix& Matrix::operator =(const Matrix& other){
    rows = other.get_rows();
    columns = other.get_columns();
    _data = other._data;
    return *this;
}

Matrix& Matrix::operator =(Matrix&& other){
    rows = other.get_rows();
    columns = other.get_columns();
    std::swap(_data, other._data);
    return *this;
}



const std::vector<double>& Matrix::operator[](size_t row) const {
    return _data[row];
}

std::vector<double>& Matrix::operator[](size_t row) {
    return _data[row];
}



size_t Matrix::get_rows() const {
    return rows;
}

size_t Matrix::get_columns() const {
    return columns;
}



Matrix Matrix::hadamard_multiply(const Matrix& lhs, const Matrix& rhs){

    Matrix result(lhs.get_rows(), lhs.get_columns());
    for (size_t i = 0; i < result.get_rows(); ++i)
        for (size_t j = 0; j < result.get_columns(); ++j)
            result[i][j] = lhs[i][j] * rhs[i][j];
    return result;
}

Matrix Matrix::transpose(const Matrix& matrix){

    Matrix result(matrix.get_columns(), matrix.get_rows());

    for (size_t i = 0; i < matrix.get_rows(); ++i)
        for (size_t j = 0; j < matrix.get_columns(); ++j)
            result[j][i] = matrix[i][j];

    return result;
}



Matrix& Matrix::operator +=(const Matrix& other){
    if (get_columns() != other.get_columns()
        || get_rows() != other.get_rows())
        throw std::invalid_argument("The dimensions of the matrices for add(+=) do not match");

    for (size_t i = 0; i < get_rows(); ++i)
        for (size_t j = 0; j < get_columns(); ++j)
            (*this)[i][j] += other[i][j];

    return *this;
}

Matrix& Matrix::operator -=(const Matrix& other){
    if (get_columns() != other.get_columns()
        || get_rows() != other.get_rows())
        throw std::invalid_argument("The dimensions of the matrices for sub(-=) do not match");

    for (size_t i = 0; i < get_rows(); ++i)
        for (size_t j = 0; j < get_columns(); ++j)
            (*this)[i][j] -= other[i][j];

    return *this;
}

Matrix& Matrix::operator *=(double coef){

    for (size_t i = 0; i < get_rows(); ++i)
        for (size_t j = 0; j < get_columns(); ++j)
            (*this)[i][j] *= coef;

    return *this;
}

Matrix& Matrix::operator /=(double coef){

    for (size_t i = 0; i < get_rows(); ++i)
        for (size_t j = 0; j < get_columns(); ++j)
            (*this)[i][j] /= coef;

    return *this;
}



Matrix operator *(const Matrix& lhs, const Matrix& rhs){
    if (lhs.get_columns() != rhs.get_rows())
        throw std::invalid_argument("The dimensions of the matrices for multiplication do not match");

    Matrix result(lhs.get_rows(), rhs.get_columns());
    for (size_t i = 0; i < result.get_rows(); ++i){
        for (size_t j = 0; j < result.get_columns(); ++j){
            result[i][j] = 0;
            for (size_t k = 0; k < lhs.get_columns(); ++k){
                result[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }
    return result;
}



Matrix operator *(double lhs, const Matrix& rhs){
    Matrix result(rhs);
    result *= lhs;
    return result;
}

Matrix operator *(const Matrix& lhs, double rhs){
    return rhs * lhs;
}

Matrix operator /(double lhs, const Matrix& rhs){
    Matrix result(rhs);
    result /= lhs;
    return result;
}

Matrix operator /(const Matrix& lhs, double rhs){
    return rhs / lhs;
}



Matrix operator -(const Matrix& lhs, const Matrix& rhs){
    if (lhs.get_rows() != rhs.get_rows()
        || lhs.get_columns() != rhs.get_columns())
        throw std::invalid_argument("The dimensions of the matrices for sub(-) do not match");

    Matrix result(lhs);
    result -= rhs;
    return result;
}

Matrix operator +(const Matrix& lhs, const Matrix& rhs){
    if (lhs.get_rows() != rhs.get_rows()
        || lhs.get_columns() != rhs.get_columns())
        throw std::invalid_argument("The dimensions of the matrices for add(+) do not match");

    Matrix result(lhs);
    result += rhs;
    return result;
}



bool operator==(const Matrix& lhs, const Matrix& rhs){
    if (lhs.get_rows() != rhs.get_rows()
        || lhs.get_columns() != rhs.get_columns())
        return false;

    for (size_t i = 0; i < lhs.get_rows(); ++i){
        if (!(lhs[i] == rhs[i]))
            return false;
    }
    return true;
}

bool operator!=(const Matrix& lhs, const Matrix& rhs){
    return !(lhs == rhs);
}



std::ostream& operator<<(std::ostream& out, const Matrix& t) {

    for (size_t i = 0; i < t.get_rows(); ++i){
        for (size_t j = 0; j < t.get_columns(); ++j)
            out << std::floor(t[i][j] * 100 + 0.5) / 100 << ' ';
        out << '\n';
    }

    return out;
}

