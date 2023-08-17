#include <cmath>
#include <stdexcept>
#include <algorithm>

#include "perceptron.h"


Perceptron::Perceptron(size_t input_dim, const std::vector<size_t>& dims) : input_dim(input_dim) {

    for (size_t i = 0; i < dims.size(); ++i){
        layers.emplace_back(input_dim, dims[i]);
        input_dim = dims[i];
    }
}

Matrix relu(const Matrix& t){
    Matrix result(t.get_rows(), t.get_columns());
    for (size_t j = 0; j < t.get_columns(); ++j)
        result[0][j] = t[0][j] > 0 ? t[0][j] : 0;
    return result;
}

Matrix relu_deriv(const Matrix& t){
    Matrix result(t.get_rows(), t.get_columns());
    for (size_t j = 0; j < t.get_columns(); ++j)
        result[0][j] = t[0][j] >= 0 ? 1 : 0;
    return result;
}

double sparse_cross_entropy(unsigned y, const Matrix& z){
    return -std::log(z[0][y]);
}

std::vector<double> sparse_cross_entropy_batch(const std::vector<unsigned>& y, const Matrix& z){
    std::vector<double> result(y.size());
    for (size_t i = 0; i < y.size(); ++i)
        result[i] = -std::log(z[i][y[i]]);
    return result;
}

std::vector<double> softmax(const std::vector<double>& t){

    double sum = 0;
    for (size_t i = 0; i < t.size(); ++i){
        sum += std::exp(t[i]);
    }

    std::vector<double> result(t.size());
    for (size_t i = 0; i < t.size(); ++i){
        result[i] = std::exp(t[i]) / sum;
    }

    return result;
}

Matrix softmax_batch(const Matrix& t){

    Matrix result(t.get_rows(), t.get_columns());

    for (size_t i = 0; i < t.get_rows(); ++i)
        result[i] = softmax(t[i]);

    return result;
}

size_t argmax(const Matrix& t){
    size_t result = 0;
    for (size_t j = 0; j < t.get_columns(); ++j)
        if (t[0][j] > t[0][result])
            result = j;
    return result;
}

std::vector<double> to_full(unsigned y, unsigned num_classes){
    std::vector<double> full(num_classes);
    full[y] = 1;
    return full;
}


Matrix to_full_batch(const std::vector<unsigned>& y, unsigned num_classes){
    Matrix full(y.size(), num_classes);
    for (size_t i = 0; i < y.size(); ++i)
        full[i][y[i]] = 1;
    return full;
}

DatasetBatch get_batch(const std::vector<DatasetItem>& dataset,
                       unsigned input_dim,
                       unsigned batch_size,
                       unsigned part) {

    DatasetBatch db {{batch_size, input_dim}, std::vector<unsigned>(batch_size)};
    for (size_t j = 0; j < batch_size; ++j){
        db.x[j] = dataset[part * batch_size + j].x;
        db.y[j] = dataset[part * batch_size + j].y;
    }
    return db;
}


Matrix Perceptron::predict(const Matrix& x){
    layers[0].input = x;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        layers[i].t = layers[i].input * layers[i].weights;

        for (size_t k = 0; k < layers[i].t.get_rows(); ++k)
            for (size_t j = 0; j < layers[i].t.get_columns(); ++j)
                layers[i].t[k][j] += layers[i].bias[0][j];

        layers[i].h = relu(layers[i].t);
        if (i + 1 != layers.size())
            layers[i + 1].input = layers[i].h;
    }
    Matrix z = softmax_batch(layers.back().t);
    return z;
}


std::vector<double> Perceptron::train(const std::vector<DatasetItem>& dataset,
                                      unsigned num_epochs,
                                      unsigned batch_size,
                                      double alpha) {

    if (dataset.size() == 0)
        throw std::invalid_argument("dataset is empty");

    std::vector<double> loss;

    std::vector<unsigned> range(dataset.size() / batch_size);
    for (size_t i = 0; i < range.size(); ++i)
        range[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());

    for (unsigned epoch = 0; epoch < num_epochs; ++epoch)
    {
        std::shuffle(range.begin(), range.end(), g);
        double error_sum = 0;
        std::cout << "---------------------" << '\n' << "epoch: " << epoch + 1 << ' ';

        for (size_t i = 0; i < range.size(); ++i)
        {
            unsigned part = range[i];
            DatasetBatch batch = get_batch(dataset, dataset[0].x.size(), batch_size, part);

            const std::vector<unsigned>& y = batch.y;
            const Matrix& x = batch.x;

            Matrix z = predict(x);

            Matrix y_full = to_full_batch(y, z.get_columns());

            std::vector<double> errors = sparse_cross_entropy_batch(y, z);
            double error_batch = 0;
            for (double error: errors)
                error_batch += error;

            loss.push_back(error_batch);
            error_sum += error_batch;

            Matrix de_dt = z - y_full;

            for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i){
                Matrix de_dw = Matrix::transpose(layers[i].input) * de_dt;

                Matrix de_db(1, de_dt.get_columns());
                for (size_t k = 0; k < de_dt.get_rows(); ++k)
                    for (size_t j = 0; j < de_dt.get_columns(); ++j)
                        de_db[0][j] += de_dt[k][j];

                Matrix de_dh = de_dt * Matrix::transpose(layers[i].weights);
                if (i > 0)
                    de_dt = Matrix::hadamard_multiply(de_dh, relu_deriv(layers[i - 1].t));

                layers[i].weights -= alpha * de_dw;
                layers[i].bias -= alpha * de_db;
            }
        }

        std::cout << "the sum of errors on this epoch: ";
        printf("%.5f\n", error_sum);
    }
    std::cout << '\n';
    return loss;
}


double Perceptron::calc_accuracy(const std::vector<DatasetItem>& dataset){
    unsigned correct = 0;
    for (const auto& item: dataset){
        unsigned y_pred = argmax(predict(item.x));
        if (item.y == y_pred)
            correct++;
    }
    return (double)correct / dataset.size();
}
