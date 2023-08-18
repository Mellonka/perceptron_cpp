#ifndef LAYER_H
#define LAYER_H

#include <random>
#include <cmath>
#include <time.h>
#include "matrix.h"


class Layer
{
private:
    const size_t input_dim;
    const size_t dim;

public:
    Layer(size_t, size_t);

    Matrix weights;
    Matrix bias;
    Matrix input;
    Matrix t;   // t = input * weights + bias
};

#endif // LAYER_H
