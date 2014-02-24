#include <opencv2/core.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>


// Do this exercise.
//
// http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html


#define DUMP(L, X) std::cerr << #L ": " #X " == " << X << std::endl;


// A test vector representing a 1650 square foot, 3-bedroom house.
//
static const cv::Mat_<double> testX(cv::Vec3d(1.0, 1650.0, 3.0));


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


// Solve a linear regression on data Y = X * THETA via normal equation.
//
class NormalizedLinearRegression
{
    cv::Mat itsSolution;
    const cv::Mat itsX;
    const cv::Mat itsY;

    // Return a copy of x after shifting columns such that x(0) is 1.
    //
    static cv::Mat makeItsX(const cv::Mat &theXs)
    {
        cv::Mat result = cv::Mat::ones(theXs.rows, 1 + theXs.cols, CV_64F);
        theXs.colRange(0, theXs.cols).copyTo(result.colRange(1, result.cols));
        return result;
    }

public:

    double hypothesis(const cv::Mat &x)
    {
        // DUMP(hypothesis, itsSolution);
        // DUMP(hypothesis, x);
        return itsSolution.dot(x);
    }

    // Return the coefficients resulting from the normal solution.
    //
    const cv::Mat &operator()(void)
    {
        if (itsSolution.empty()) {
            const cv::Mat xTx = itsX.t() * itsX;
            itsSolution = xTx.inv() * itsX.t() * itsY;
        }
        return itsSolution;
    }

    NormalizedLinearRegression(const cv::Mat &theXs, const cv::Mat &theYs)
        : itsSolution()
        , itsX(makeItsX(theXs))
        , itsY(theYs.clone())
    {}
};


// Demonstrate gradient descent for solving linear regression problems.
//
// Be careful not to alias itsPriorTheta and itsTheta! =tbl
//
class GradientDescent
{
    const double itsAlpha;
    cv::Mat itsX;
    const cv::Mat itsY;
    cv::Mat itsMean;
    cv::Mat itsStdDevInv;
    cv::Mat itsPriorTheta;
    cv::Mat itsTheta;
    size_t itsIterationCount;

    // Update itsTheta by descending once along the steepest gradient.
    //
    void descend(void)
    {
        itsTheta.copyTo(itsPriorTheta);
        const cv::Mat delta = itsX.t() * (itsX * itsTheta - itsY) / itsY.rows;
        itsTheta = itsTheta - (delta * itsAlpha);
        ++itsIterationCount;
    }

    // Return true iff all components of itsPriorTheta and itsTheta differ
    // by less than epsilon.
    //
    bool epsilonSatisfied(double epsilon)
    {
        cv::Mat d; cv::absdiff(itsTheta, itsPriorTheta, d);
        double maximumValue = 0.0;
        cv::minMaxLoc(d, NULL, &maximumValue, NULL, NULL);
        return maximumValue < epsilon;
    }

    // Return a copy of x after shifting columns x(i) to x(i=1) such that
    // x(0) is 1.
    //
    static cv::Mat makeItsX(const cv::Mat &theXs)
    {
        cv::Mat result = cv::Mat::ones(theXs.rows, 1 + theXs.cols, CV_64F);
        theXs.colRange(0, theXs.cols).copyTo(result.colRange(1, result.cols));
        return result;
    }

    // Return the vector scaled from the vector x.
    //
    cv::Mat scale(const cv::Mat &x) { return itsStdDevInv * (x - itsMean); }

public:

    double hypothesis(const cv::Mat &x)
    {
        // DUMP(hypothesis, itsTheta);
        // DUMP(hypothesis, x);
        return itsTheta.dot(scale(x));
    }

    // Return the theta resulting from n descents.
    //
    const cv::Mat theta(int n)
    {
        if (n > itsIterationCount) {
            size_t rest = n - itsIterationCount;
            while(rest--) descend();
        }
        return itsTheta;
    }

    // Return the Theta vector after meeting tc.
    //
    const cv::Mat operator()(const cv::TermCriteria &tc)
    {
        const int useCount = tc.type & cv::TermCriteria::COUNT;
        const int useEpsilon = tc.type & cv::TermCriteria::EPS;
        if (useCount && useEpsilon) {
            bool done = true;
            do {
                descend();
                done = itsIterationCount >= tc.maxCount
                    || epsilonSatisfied(tc.epsilon);
            } while (!done);
        } else if (useEpsilon) {
            do { descend(); } while (!epsilonSatisfied(tc.epsilon));
        } else if (useCount) {
            theta(tc.maxCount);
        }
        return itsTheta;
    }

    // Return the number of iterations this has run.
    //
    size_t count(void) { return itsIterationCount; }

    // Initialize gradient descent with learning rate alpha, feature
    // vectors theXs and result vectors theYs.  Start with theta [0,...].
    //
    GradientDescent(double alpha, const cv::Mat &theXs, const cv::Mat &theYs)
        : itsAlpha(alpha)
        , itsX(makeItsX(theXs))
        , itsY(theYs.clone())
        , itsMean(cv::Mat::zeros(itsX.cols, 1, CV_64F))
        , itsStdDevInv(cv::Mat::zeros(itsX.cols, itsX.cols, CV_64F))
        , itsPriorTheta(cv::Mat::zeros(itsX.cols, 1, CV_64F))
        , itsTheta(cv::Mat::zeros(itsX.cols, 1, CV_64F))
        , itsIterationCount(0)
    {
        itsMean.at<double>(0, 0) = 0.0;
        itsStdDevInv.at<double>(0, 0) = 1.0;
        for (int i = 1; i < itsX.cols; ++i) {
            cv::Scalar mean, dev;
            cv::meanStdDev(itsX.col(i), mean, dev);
            itsMean.at<double>(0, i) = mean.val[0];
            itsStdDevInv.at<double>(i, i) = 1.0 / dev.val[0];
        }
        for (int i = 0; i < itsX.rows; ++i) {
            const cv::Mat scaled = scale(itsX.row(i).t()).t();
            scaled.copyTo(itsX.row(i));
        }
    }
};


// Show on os the resulting equation with coefficients theta.
//
static void showResult(std::ostream &os, const cv::Mat &theta)
{
    os << "y = " << std::fixed << std::setprecision(4)
       << theta.at<double>(0, 0);
    for (int i = 1; i < theta.rows; ++i) {
        double t = theta.at<double>(i, 0);
        const char *sign = " + ";
        if (t < 0) {
            sign = " - ";
            t = 0.0 - t;
        }
        os << sign << std::fixed << std::setprecision(4) << t << " * x" << i;
    }
    os << std::endl;
}


// Report on os gradient descent on alpha, xv, and yv until some
// termination criteria are met.
//
static void withTermCriteria(std::ostream &os, double alpha,
                             const cv::Mat &xs, const cv::Mat &ys)
{
    static const int criteria
        = cv::TermCriteria::COUNT | cv::TermCriteria::EPS;
    static const double epsilon = 0.0000000000005;
    static const int iterations = 100;
    static const cv::TermCriteria tc(criteria, iterations, epsilon);
    os << std::endl << "Now use TermCriteria: "
       << tc.maxCount << " iterations or an epsilon of "
       << std::scientific << tc.epsilon
       << std::endl;
    GradientDescent gd(alpha, xs, ys);
    const cv::Mat theta = gd(tc);
    os << "TermCriteria met after " << gd.count() << " iterations."
       << std::endl << std::endl;
    showResult(os, theta);
    const double gdHypo = gd.hypothesis(testX);
    DUMP(main, gdHypo);
}


int main(int ac, const char *av[])
{
    if (ac == 3) {
        std::ifstream xis(av[1]), yis(av[2]);
        double y; std::vector<double> yv; while (yis >> y) yv.push_back(y);
        cv::Mat_<double> theYs(yv);
        cv::Mat_<double> theXs(theYs.rows, 2, CV_64F);
        for (int i = 0; i < yv.size(); ++i) xis >> theXs(i, 0) >> theXs(i, 1);
        static const double alpha = 1.0;
        GradientDescent gd(alpha, theXs, theYs);
        const cv::Mat theta1 = gd.theta(1);
        DUMP(main, theta1);
        withTermCriteria(std::cout, alpha, theXs, theYs);
        NormalizedLinearRegression normal(theXs, theYs);
        showResult(std::cout, normal());
        DUMP(main, testX);
        const double normalHypo = normal.hypothesis(testX);
        DUMP(main, normalHypo);
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
