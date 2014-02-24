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


// Solve a linear regression on data Y = X * THETA via the normal equation.
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
        cv::Mat result = cv::Mat::ones(theXs.rows, 1 + theXs.cols, CV_32FC1);
        theXs.colRange(0, theXs.cols).copyTo(result.colRange(1, result.cols));
        return result;
    }

public:

    // Return the result of applying this solution to x.
    //
    float hypothesis(const cv::Mat &x) {
        const cv::Mat shifted = makeItsX(x.t());
        return itsSolution.dot(shifted.t());
    }

    // Return the coefficients resulting from the normal solution.  Use
    // DECOMP_SVD to preserve the 1.0 component.  DECOMP_CHOLESKY and
    // DECOMP_LU both lose some precision.
    //
    // This is just an optimized coding of the more explicit code below.
    //            const cv::Mat xTx = itsX.t() * itsX;
    //            itsSolution = xTx.inv() * itsX.t() * itsY;
    //
    const cv::Mat &operator()(void)
    {
        static const bool transposeLeft = true;
        if (itsSolution.empty()) {
            cv::Mat xTxI; cv::mulTransposed(itsX, xTxI, transposeLeft);
            cv::invert(xTxI, xTxI, cv::DECOMP_SVD);
            itsSolution = xTxI * itsX.t() * itsY;
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
    const float itsAlpha;
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
        cv::Mat result = cv::Mat::ones(theXs.rows, 1 + theXs.cols, CV_32FC1);
        theXs.colRange(0, theXs.cols).copyTo(result.colRange(1, result.cols));
        return result;
    }

    // Return the vector scaled from x by itsStdDevInv and itsMean
    // calculated from the distribution of all itsX.
    //
    cv::Mat scale(const cv::Mat &x) { return itsStdDevInv * (x - itsMean); }

public:

    // Return the result of applying the current solution to x.
    //
    float hypothesis(const cv::Mat &x) {
        const cv::Mat shifted = makeItsX(x.t());
        return itsTheta.dot(scale(shifted.t()));
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
    // vectors theXs and result vectors theYs.  Scale theXs and start with
    // theta [0,...].
    //
    GradientDescent(float alpha, const cv::Mat &theXs, const cv::Mat &theYs)
        : itsAlpha(alpha)
        , itsX(makeItsX(theXs))
        , itsY(theYs.clone())
        , itsMean(cv::Mat::zeros(itsX.cols, 1, CV_32FC1))
        , itsStdDevInv(cv::Mat::zeros(itsX.cols, itsX.cols, CV_32FC1))
        , itsPriorTheta(cv::Mat::zeros(itsX.cols, 1, CV_32FC1))
        , itsTheta(cv::Mat::zeros(itsX.cols, 1, CV_32FC1))
        , itsIterationCount(0)
    {
        itsMean.at<float>(0, 0) = 0.0;
        itsStdDevInv.at<float>(0, 0) = 1.0;
        for (int i = 1; i < itsX.cols; ++i) {
            cv::Scalar mean, dev;
            cv::meanStdDev(itsX.col(i), mean, dev);
            itsMean.at<float>(0, i) = mean.val[0];
            itsStdDevInv.at<float>(i, i) = 1.0 / dev.val[0];
        }
        for (int i = 0; i < itsX.rows; ++i) {
            const cv::Mat scaled = scale(itsX.row(i).t()).t();
            scaled.copyTo(itsX.row(i));
        }
    }
};


// Show on os label and the resulting equation with coefficients theta.
//
static void showResult(std::ostream &os, const char *label,
                       const cv::Mat &theta)
{
    os << label << ": y = " << std::fixed << std::setprecision(4)
       << theta.at<float>(0, 0);
    for (int i = 1; i < theta.rows; ++i) {
        float t = theta.at<float>(i, 0);
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
static void withTermCriteria(std::ostream &os, float alpha,
                             const cv::Mat &xs, const cv::Mat &ys)
{
    static const int criteria
        = cv::TermCriteria::COUNT | cv::TermCriteria::EPS;
    static const double epsilon = 0.000000000001;
    static const int iterations = 1000;
    static const cv::TermCriteria tc(criteria, iterations, epsilon);
    os << std::endl << "TermCriteria: " << tc.maxCount
       << " iterations or epsilon of " << std::scientific
       << tc.epsilon << std::endl;
    GradientDescent gd(alpha, xs, ys);
    const cv::Mat theta = gd(tc);
    os << "TermCriteria at alpha " << std::fixed << alpha
       << " met after " << gd.count() << " iterations." << std::endl;
    showResult(os, "Scaled GradientDescent", theta);
    DUMP(withTermCriteria, gd.hypothesis(testX));
}


int main(int ac, const char *av[])
{
    if (ac == 3) {
        std::ifstream xis(av[1]), yis(av[2]);
        float y; std::vector<float> yv; while (yis >> y) yv.push_back(y);
        cv::Mat_<float> theYs(yv), theXs(theYs.rows, 2, CV_32FC1);
        for (int i = 0; i < yv.size(); ++i) xis >> theXs(i, 0) >> theXs(i, 1);
        NormalizedLinearRegression normal(theXs, theYs);
        std::cout << std::endl;
        showResult(std::cout, "NormalizedLinearRegression", normal());
        DUMP(main, testX);
        DUMP(main, normal.hypothesis(testX));
        for (float alpha = 0.1; alpha < 1.6; alpha += 0.1) {
            withTermCriteria(std::cout, alpha, theXs, theYs);
        }
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
