#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>

#include "perceptron.h"


void iris_dataset(){
    std::ifstream file("\\Iris.txt");
    std::unordered_map<std::string, unsigned> label_to_y = {
        {"Iris-setosa", 0},
        {"Iris-virginica", 1},
        {"Iris-versicolor", 2},
    };

    std::unordered_map<unsigned, std::string> y_to_label = {
        {0, "Iris-setosa"},
        {1, "Iris-virginica"},
        {2, "Iris-versicolor"},
    };

    std::vector<DatasetItem> dataset;
    int id;
    std::string label;

    while(file >> id) {
        std::vector<double> x(4);
        file >> x[0] >> x[1] >> x[2] >> x[3] >> label;
        dataset.push_back({x, label_to_y[label]});
    }

    Perceptron perc(4, {10, 3});
    perc.train(dataset, 500, 15, 0.001);

    std::cout << "accuracy: " << perc.calc_accuracy(dataset) << '\n';
    std::cout << "predicted: " << y_to_label[argmax(perc.predict(std::vector<double>{4.9, 3.1, 1.5, 0.1}))] << "| expected: Iris-setosa" << '\n';
    std::cout << "predicted: " << y_to_label[argmax(perc.predict(std::vector<double>{6.5, 3.0, 5.2, 2.0}))] << "| expected: Iris-virginica" << '\n';
}


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
































