#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include "layer.h"


struct DatasetItem{
    std::vector<double> x;
    unsigned y;
};

struct DatasetBatch {
    Matrix x;
    std::vector<unsigned> y;
};

class Perceptron {
private:
    const size_t input_dim;
    std::vector<Layer> layers;

public:
    Perceptron(size_t input_dim, const std::vector<size_t>& dims);

    std::vector<double> train(const std::vector<DatasetItem>&,
                              unsigned = 100,
                              unsigned = 1,
                              double = 0.0001);

    double calc_accuracy(const std::vector<DatasetItem>&);

    Matrix predict(const Matrix& x);
};


Matrix relu(const Matrix&);
Matrix relu_deriv(const Matrix&);

double sparse_cross_entropy(unsigned y, const Matrix& z);
std::vector<double> sparse_cross_entropy_batch(const std::vector<unsigned>& y, const Matrix& z);

std::vector<double> softmax(const std::vector<double>&);
Matrix softmax_batch(const Matrix&);

std::vector<double> to_full(unsigned y, unsigned num_classes);
Matrix to_full_batch(const std::vector<unsigned>& y, unsigned num_classes);

DatasetBatch get_batch(const std::vector<DatasetItem>&,
                       unsigned input_dim,
                       unsigned batch_size,
                       unsigned part);

size_t argmax(const Matrix&);

#endif // PERCEPTRON_H
