#include "layer.h"


Layer::Layer(size_t input_dim, size_t dim)
    : input_dim(input_dim),
      dim(dim),
      weights(input_dim, dim),
      bias(1, dim),
      input(1, input_dim),
      t(1, dim),
      h(1, dim) {

    srand(time(NULL));

    for (size_t i = 0; i < input_dim; ++i){
        for (size_t j = 0; j < dim; ++j){
            weights[i][j] = (double)rand() / RAND_MAX;
        }
    }

    for (size_t j = 0; j < dim; ++j){
        bias[0][j] = (double)rand() / RAND_MAX;
    }
}


