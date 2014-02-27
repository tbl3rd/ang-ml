#include <opencv2/core.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>


// Do this exercise.
//
// http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html


#define DUMP(L, X) std::cerr << #L ": " #X " == " << X << std::endl;


// A test vector representing a 1650 square foot, 3-bedroom house.
//
static const cv::Mat_<float> testX(cv::Vec2f(1650.0, 3.0));


static void showUsage(const char *av0)
{
    std::cout << av0 << ": Demonstrate multivariate linear regression."
              << std::endl << std::endl
              << "Usage: " << av0 << " <xs> <ys>" << std::endl << std::endl
              << "Where: <xs> contains the training features, 1 per line."
              << std::endl
              << "       <ys> contains the training labels, 1 per line."
              << std::endl << std::endl;
}


// Return the result of applying f to each row of m.
//
template <typename T> static T
applyToRow(const T &m, T f(const T &))
{
    T result(m.size(), m.type());
    for (int i = 0; i < m.rows; ++i) f(m.row(i)).copyTo(result.row(i));
    return result;
}


int main(int ac, const char *av[])
{
    if (ac == 3) {
        std::ifstream xis(av[1]), yis(av[2]);
        float y; std::vector<float> yv; while (yis >> y) yv.push_back(y);
        cv::Mat_<float> theYs(yv), theXs(theYs.rows, 2, CV_32FC1);
        for (int i = 0; i < yv.size(); ++i) xis >> theXs(i, 0) >> theXs(i, 1);
        DUMP(main, theYs);
        DUMP(main, theXs);
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
