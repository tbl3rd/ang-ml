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


class LogisticalRegression
{
    const cv::Mat itsX;
    const cv::Mat itsY;
    cv::Mat itsTheta;

    // Return a copy of x after shifting columns such that x(0) is 1.
    //
    static cv::Mat makeItsX(const cv::Mat &theXs)
    {
        cv::Mat result = cv::Mat::ones(theXs.rows, 1 + theXs.cols, CV_32FC1);
        theXs.colRange(0, theXs.cols).copyTo(result.colRange(1, result.cols));
        return result;
    }

    // Return the logistical cost J(theta) of the model theta applied to x
    // compared to the expected y.
    //
    // (Not related to the Jacobian, but we do use the gradient below.)
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
        const cv::Mat notY = 1.0 - y;
        double result = 0.0 - y.dot(hyTheta) - notY.dot(notHyTheta);
        result /= x.rows;
        return result;
    }

    // Return the gradient vector of J(theta).
    //
    static cv::Mat gradient(const cv::Mat_<float> &theta,
                            const cv::Mat_<float> &x,
                            const cv::Mat_<float> &y)
    {
        cv::Mat result = 0.0 - x * theta;
        cv::exp(result, result);
        result += 1.0;
        cv::divide(1.0, result, result);
        result -= y;
        result = result.t() * x;
        return result.t();
    }

    // Return the Hessian matrix for J(theta).
    //
    static cv::Mat hessian(const cv::Mat_<float> &theta,
                           const cv::Mat_<float> &x)
    {
        static const cv::Mat noDelta;
        static const bool mTm = false;
        cv::Mat result = cv::Mat::zeros(theta.rows, theta.rows, theta.type());
        for (int i = 0; i < x.rows; ++i) {
            const cv::Mat xi = x.row(i).t();
            const double exponent = 0.0 - theta.dot(xi);
            const double hTheta = 1.0 / (1.0 + std::exp(exponent));
            const double scale = hTheta * (1.0 - hTheta);
            const cv::Mat term(theta.rows, theta.rows, theta.type());
            cv::mulTransposed(xi, term, mTm, noDelta, scale);
            result += term;
        }
        return result;
    }

public:

    LogisticalRegression(const cv::Mat &x, const cv::Mat &y)
        : itsX(makeItsX(x))
        , itsY(y.clone())
        , itsTheta(cv::Mat::zeros(itsX.cols, 1, itsX.type()))
    {
        DUMP(LogisticalRegression, itsTheta);
        DUMP(LogisticalRegression, cost(itsTheta, itsX, itsY));
        DUMP(LogisticalRegression, gradient(itsTheta, itsX, itsY));
        DUMP(LogisticalRegression, hessian(itsTheta, itsX));
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
