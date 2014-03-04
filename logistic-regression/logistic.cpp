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


class LogisticalRegression
{
    cv::Mat itsTheta;
    const cv::Mat itsX;
    const cv::Mat itsY;

    // Return a copy of x after shifting columns such that x(0) is 1.
    //
    static cv::Mat makeItsX(const cv::Mat &theXs)
    {
        cv::Mat result = cv::Mat::ones(theXs.rows, 1 + theXs.cols, CV_32FC1);
        theXs.colRange(0, theXs.cols).copyTo(result.colRange(1, result.cols));
        return result;
    }

    // Return the logistical cost of the model theta applied to x compared
    // to the expected y.
    //
    static float cost(const cv::Mat_<float> &theta,
                      const cv::Mat_<float> &x,
                      const cv::Mat_<float> &y)
    {
        cv::Mat hyTheta = 0.0 - x * theta;
        cv::exp(hyTheta, hyTheta);
        hyTheta += 1.0;
        cv::divide(1.0, hyTheta, hyTheta);
        cv::Mat notHyTheta = 1.0 - hyTheta;
        cv::log(hyTheta, hyTheta);
        cv::log(notHyTheta, notHyTheta);
        double result = 0.0 - y.dot(hyTheta) - (1.0 - y).dot(notHyTheta);
        result /= y.rows;
        return result;
    }

public:

    LogisticalRegression(const cv::Mat &x, const cv::Mat &y)
        : itsTheta(1 + x.cols, 1, y.type())
        , itsX(makeItsX(x))
        , itsY(y.clone())
    {
        DUMP(LogisticalRegression, cost(itsTheta, itsX, itsY));
    }
};


int main(int ac, const char *av[])
{
    if (ac == 3) {
        std::ifstream xis(av[1]), yis(av[2]);
        float y; std::vector<float> yv; while (yis >> y) yv.push_back(y);
        cv::Mat_<float> theYs(yv), theXs(theYs.rows, 2, CV_32FC1);
        for (int i = 0; i < yv.size(); ++i) xis >> theXs(i, 0) >> theXs(i, 1);
        DUMP(CxR, theYs.size());
        DUMP(CxR, theXs.size());
        LogisticalRegression lr(theXs, theYs);
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
