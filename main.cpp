#include <iostream>
#include "perceptron.h"

int main(){

    std::vector<double> test1 = {1, 0};
    std::vector<double> test2 = {0, 1};
    std::vector<double> test3 = {1, 1};
    std::vector<double> test4 = {0, 0};


    std::vector<DatasetItem> datasetXOR = {
        {test1, 1},
        {test2, 1},
        {test3, 0},
        {test4, 0}
    };

    std::vector<DatasetItem> datasetAND = {
        {test1, 0},
        {test2, 0},
        {test3, 1},
        {test4, 0}
    };

    std::vector<DatasetItem> datasetOR = {
        {test1, 1},
        {test2, 1},
        {test3, 1},
        {test4, 0}
    };

    Perceptron perc(2, {3, 2});
    perc.train(datasetXOR, 500, 1, 0.1);

    std::cout << "accuracy: " << perc.calc_accuracy(datasetXOR) << '\n';
    std::cout << test1[0] << ' ' << test1[1] << " - " << argmax(perc.predict(test1)) << '\n';
    std::cout << test2[0] << ' ' << test2[1] << " - " << argmax(perc.predict(test2)) << '\n';
    std::cout << test3[0] << ' ' << test3[1] << " - " << argmax(perc.predict(test3)) << '\n';
    std::cout << test4[0] << ' ' << test4[1] << " - " << argmax(perc.predict(test4)) << '\n';
}
