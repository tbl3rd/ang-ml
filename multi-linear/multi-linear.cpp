#include <opencv2/core.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>


// Do this exercise.
//
// http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html


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
        cv::Mat result = cv::Mat::ones(theXs.rows, 1 + theXs.cols, CV_32F);
        theXs.colRange(0, theXs.cols).copyTo(result.colRange(1, result.cols));
        return result;
    }

public:

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
    const float itsAlpha;
    cv::Mat itsX;
    const cv::Mat itsY;
    const cv::Mat itsScales;
    cv::Mat itsPriorTheta;
    cv::Mat itsTheta;
    size_t itsIterationCount;

    // Update itsTheta by descending once along the steepest gradient.
    //
    void descend(void)
    {
        itsTheta.copyTo(itsPriorTheta);
        const cv::Mat delta
            = itsX.t() * (itsX * itsTheta - itsY) / itsY.rows;
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
    static cv::Mat makeItsXs(const cv::Mat &theXs)
    {
        cv::Mat result = cv::Mat::ones(theXs.rows, 1 + theXs.cols, CV_32F);
        theXs.colRange(0, theXs.cols).copyTo(result.colRange(1, result.cols));
        return result;
    }

    // Just return a copy of y.
    //
    static cv::Mat makeItsYs(const cv::Mat &theYs)
    {
        cv::Mat result;
        theYs.copyTo(result);
        return result;
    }

    // Compute a scaling factor for each dimension (column) of itsX and
    // return it as a diagonal matrix after multiplying itsX by the
    // result.
    //
    static cv::Mat scaleItsXs(cv::Mat &itsX)
    {
        static const double epsilon = std::numeric_limits<float>::epsilon();
        cv::Mat result = cv::Mat::zeros(itsX.cols, itsX.cols, CV_32F);
        for (int i = 0; i < result.cols; ++i) {
            cv::Mat_<float> x = itsX.col(i);
            double maximum, minimum;
            cv::minMaxLoc(x, &minimum, &maximum);
            const double difference = maximum - minimum;
            const double scale = difference > epsilon ? difference : 1.0;
            result.at<float>(i, i) = 1.0 / scale;
        }
        itsX *= result;
        return result;
    }

public:

    // Return the scaled theta resulting from n descents.
    //
    const cv::Mat theta(int n)
    {
        if (n > itsIterationCount) {
            size_t rest = n - itsIterationCount;
            while(rest--) descend();
        }
        DUMP(theta, itsScales);
        DUMP(theta, itsTheta);
        return itsScales * itsTheta;
    }

    // Return the scaled Theta vector after meeting tc.
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
        return itsScales * itsTheta;
    }

    // Return the number of iterations this has run.
    //
    size_t count(void) { return itsIterationCount; }

    cv::Mat normal(void)
    {
        const cv::Mat xtx = itsX.t() * itsX;
        const cv::Mat result = xtx.inv() * itsX.t() * itsY;
        DUMP(normal, result);
        return result;
    }

    // Initialize gradient descent with learning rate alpha, feature
    // vectors theXs and result vectors theYs.  Start with theta [0,...].
    //
    GradientDescent(float alpha, const cv::Mat &theXs, const cv::Mat &theYs)
        : itsAlpha(alpha)
        , itsX(makeItsXs(theXs))
        , itsY(makeItsYs(theYs))
        , itsScales(scaleItsXs(itsX))
        , itsPriorTheta(cv::Mat::zeros(itsX.cols, 1, CV_32F))
        , itsTheta(cv::Mat::zeros(itsX.cols, 1, CV_32F))
        , itsIterationCount(0)
    {
        // DUMP(GradientDescent, itsY.size());
        // DUMP(GradientDescent, itsY);
        // DUMP(GradientDescent, itsX.size());
        // DUMP(GradientDescent, itsX);
        // DUMP(GradientDescent, itsTheta.size());
        // DUMP(GradientDescent, itsTheta);
        // DUMP(GradientDescent, itsScales.size());
        // DUMP(GradientDescent, itsScales);
        // DUMP(GradientDescent, itsScales.diag());
        // DUMP(GradientDescent, itsScales.inv());
    }
};


// Show on os the resulting equation with coefficients theta.
//
static void showResult(std::ostream &os, const cv::Mat &theta)
{
    os << "y = " << std::fixed << std::setprecision(4) << theta.at<float>(0, 0);
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
    static const float epsilon = 0.0001;
    static const int iterations = 100;
    static const cv::TermCriteria tc(criteria, iterations, epsilon);
    os << std::endl << "Now use TermCriteria: "
       << tc.maxCount << " iterations or an epsilon of "
       << std::scientific << tc.epsilon
       << std::endl;
    GradientDescent gdTc(alpha, xs, ys);
    const cv::Mat theta = gdTc(tc);
    os << "TermCriteria met after " << gdTc.count() << " iterations."
       << std::endl << std::endl;
    showResult(os, theta);
}


int main(int ac, const char *av[])
{
    if (ac == 3) {
        std::ifstream xis(av[1]), yis(av[2]);
        float y; std::vector<float> yv; while (yis >> y) yv.push_back(y);
        cv::Mat_<float> theYs(yv);
        cv::Mat_<float> theXs(theYs.rows, 2, CV_32F);
        for (int i = 0; i < yv.size(); ++i) xis >> theXs(i, 0) >> theXs(i, 1);
        NormalizedLinearRegression normal(theXs, theYs);
        DUMP(main, normal());
        showResult(std::cout, normal());
        static const float alpha = 1.0;
        GradientDescent gd(alpha, theXs, theYs);
        const cv::Mat theta1 = gd.theta(1);
        DUMP(main, theta1);
        withTermCriteria(std::cout, alpha, theXs, theYs);
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
