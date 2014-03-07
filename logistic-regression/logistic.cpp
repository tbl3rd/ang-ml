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


// A test vector representing test scores of 20 and 80.
//
static const cv::Mat_<float> testX(cv::Vec2f(20.0, 80.0));


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


// Run unscaled logistical regression to classify features x into discrete
// y of either 0 or 1.
//
class LogisticalRegression
{
    const cv::Mat itsX;
    const cv::Mat itsY;
    cv::Mat itsTheta;
    cv::Mat itsPriorTheta;
    int itsIterationCount;

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
    static float costTheta(const cv::Mat_<float> &theta,
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

    // Return the gradient vector of costTheta(), J(theta).
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

    // Return the Hessian matrix for cost(), J(theta).
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

    // Apply one iteration of Newton's method to the model.
    //
    void newton(void)
    {
        itsTheta.copyTo(itsPriorTheta);
        const cv::Mat h = hessian(itsTheta, itsX);
        const cv::Mat g = gradient(itsTheta, itsX, itsY);
        itsTheta = itsPriorTheta - h.inv() * g;
    }

public:

    // Return the result of applying the current solution to x.
    //
    float hypothesis(const cv::Mat &x) {
        const cv::Mat shifted = makeItsX(x.t());
        const double exponent = 0.0 - itsTheta.dot(shifted.t());
        const double result = 1.0 / (1.0 + std::exp(exponent));
        return result;
    }

    // Return the cost, J(theta), evaluated at the current theta vector.
    //
    float cost(void) { return costTheta(itsTheta, itsX, itsY); }

    // Return the theta resulting from n descents.
    //
    const cv::Mat theta(int n)
    {
        if (n > itsIterationCount) {
            int rest = n - itsIterationCount;
            while (rest--) newton();
        }
        return itsTheta;
    }

    // Return the difference between the current value of theta and its
    // prior value.
    //
    double epsilon(void)
    {
        cv::Mat d; cv::absdiff(itsTheta, itsPriorTheta, d);
        double result = 0.0;
        cv::minMaxLoc(d, NULL, &result, NULL, NULL);
        return result;
    }

    // Return the Theta vector after meeting termination criteria tc.
    //
    const cv::Mat operator()(const cv::TermCriteria &tc)
    {
        const int useCount = tc.type & cv::TermCriteria::COUNT;
        const int useEpsilon = tc.type & cv::TermCriteria::EPS;
        if (useCount && useEpsilon) {
            bool done = true;
            do {
                newton();
                done = itsIterationCount >= tc.maxCount
                    || epsilon() < tc.epsilon;
            } while (!done);
        } else if (useEpsilon) {
            do { newton(); } while (!epsilon() < tc.epsilon);
        } else if (useCount) {
            theta(tc.maxCount);
        }
        return itsTheta;
    }

    LogisticalRegression(const cv::Mat &x, const cv::Mat &y)
        : itsX(makeItsX(x))
        , itsY(y.clone())
        , itsTheta(cv::Mat::zeros(itsX.cols, 1, itsX.type()))
        , itsPriorTheta(itsTheta.clone())
    {
        // DUMP(LogisticalRegression, itsTheta);
        // DUMP(LogisticalRegression, costTheta(itsTheta, itsX, itsY));
        // DUMP(LogisticalRegression, gradient(itsTheta, itsX, itsY));
        // DUMP(LogisticalRegression, hessian(itsTheta, itsX));
    }
};


// Tabulate the data from lr on os.
//
#define SHOW(X) std::fixed << std::setprecision(8) << std::setw(12) << X
static void showData(std::ostream &os, LogisticalRegression &lr)
{
    os << std::endl
       << " N " << SHOW("COST") << SHOW("EPSILON") << "    THETA" << std::endl
       << "---" << SHOW("----") << SHOW("-------") << "    -----" << std::endl;
    for (int i = 0; i < 9; ++i) {
        const cv::Mat theta = lr.theta(i);
        os << " " << i << " "
           << SHOW(lr.cost()) << SHOW(lr.epsilon()) << "    " << theta
           << std::endl;
    }
}
#undef SHOW


int main(int ac, const char *av[])
{
    if (ac == 3) {
        std::ifstream xis(av[1]), yis(av[2]);
        float y; std::vector<float> yv; while (yis >> y) yv.push_back(y);
        cv::Mat_<float> theYs(yv), theXs(theYs.rows, 2, CV_32FC1);
        for (int i = 0; i < yv.size(); ++i) xis >> theXs(i, 0) >> theXs(i, 1);
        LogisticalRegression lr(theXs, theYs);
        showData(std::cout, lr);
        const float testY = lr.hypothesis(testX);
        std::cout << std::endl
                  << "Probability " << std::setprecision(2) << testY * 100.0
                  << "% that student with test scores " << testX
                  << " will be admitted." << std::endl;
        std::cout << "Probability " << std::setprecision(2)
                  << (1.0 - testY) * 100.0
                  << "% that student with test scores " << testX
                  << " will NOT be admitted." << std::endl;
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
