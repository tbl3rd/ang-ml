#include <opencv2/core.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>


// Do this exercise.
//
// http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html


#define DUMP(L, X) std::cerr << #L ": " #X " == " << X << std::endl;


static void showUsage(const char *av0)
{
    std::cout << av0 << ": Demonstrate linear regression."
              << std::endl << std::endl
              << "Usage: " << av0 << " <xs> <ys>" << std::endl << std::endl
              << "Where: <xs> contains the training features, 1 per line."
              << std::endl
              << "       <ys> contains the training labels, 1 per line."
              << std::endl << std::endl;
}


// Solve a linear regression on data Y = X * THETA via the regularized
// normal equation.
//
class RegularizedNormalEquation
{
    const cv::Mat &itsX;
    const cv::Mat &itsY;
    const float itsLambda;
    cv::Mat itsTheta;

public:

    // Return the coefficients resulting from the normal equation.
    //
    const cv::Mat &operator()(void)
    {
        if (itsTheta.empty()) {
            cv::Mat lambda = cv::Mat::eye(itsX.cols, itsX.cols, itsX.type());
            lambda *= itsLambda;
            lambda.at<float>(0, 0) = 0.0;
            lambda += itsX.t() * itsX;
            itsTheta = lambda.inv() * itsX.t() * itsY;
        }
        return itsTheta;
    }

    // Construct the normal equation solution of Y = X * THETA with
    // regularization parameter lambda
    //
    RegularizedNormalEquation(const cv::Mat &X, const cv::Mat &Y, float lambda)
        : itsX(X)
        , itsY(Y)
        , itsLambda(lambda)
        , itsTheta()
    {}
};


#define SHOW(X) std::fixed << std::setprecision(4) << std::setw(14) << X
static void showResults(std::ostream &os, const cv::Mat &x, const cv::Mat &y)
{
    static const float lambda[] = { 0.0, 1.0, 10.0 };
    static const int lambdaCount = sizeof lambda / sizeof lambda[0];
    std::vector<cv::Mat> thetas;
    os << std::endl << SHOW("RESULTS");
    for (int i = 0; i < lambdaCount; ++i) {
        std::stringstream label; label << "Lambda=" << lambda[i];
        os << SHOW(label.str());
        RegularizedNormalEquation rne(x, y, lambda[i]);
        thetas.push_back(rne().clone());
    }
    os << std::endl << std::endl;
    os << SHOW("L2 Norm");
    for (int i = 0; i < lambdaCount; ++i) {
        os << SHOW(cv::norm(thetas[i]));
    }
    os << std::endl << std::endl;
    for (int i = 0; i < thetas[0].rows; ++i) {
        std::stringstream label; label << "Theta" << i;
        os << SHOW(label.str());
        for (int j = 0; j < lambdaCount; ++j) {
            os << SHOW(thetas[j].at<float>(i, 0));
        }
        os << std::endl;
    }
    os << std::endl;
}
#undef SHOW


int main(int ac, const char *av[])
{
    if (ac == 3) {
        std::ifstream xis(av[1]), yis(av[2]);
        float y; std::vector<float> yv; while (yis >> y) yv.push_back(y);
        static const int dim = 6;
        cv::Mat_<float> theYs(yv), theXs(theYs.rows, dim, CV_32FC1);
        for (int i = 0; i < yv.size(); ++i) {
            float x; xis >> x;
            for (int n = 0; n < dim; ++n) theXs(i, n) = std::pow(x, n);
        }
        showResults(std::cout, theXs, theYs);
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
