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


// Demonstrate gradient descent for solving linear regression problems.
//
// Be careful not to alias itsPriorTheta and itsTheta! =tbl
//
class GradientDescent
{
    const float itsAlpha;
    cv::Mat itsXs;
    const cv::Mat itsYs;
    const cv::Mat itsScales;
    cv::Mat itsPriorTheta;
    cv::Mat itsTheta;
    size_t itsIterationCount;

    // Return the delta vector for theta after evaluating x against the
    // current theta to estimate y.  This is the "cost function" J(theta)
    // where itsTheta.dot(x) is the hypothesis h(x) for a linear regression
    // model with vector theta.
    //
    // Some terms are commuted because C++'s OO dispatch works only if the
    // matrix (this) factor is on the left side of expressions.
    //
    cv::Mat delta(void)
    {
        const int m = itsYs.rows;
        cv::Mat sum = cv::Mat::zeros(1, itsXs.cols, CV_32F);
        for (int i = 0; i < m; ++i) {
            const cv::Mat x = itsXs.row(i);
            const float y = itsYs.at<float>(i);
            const float h = itsTheta.dot(x);
            sum += x * (h - y);
        }
        return sum / m;
    }

    // Update itsTheta by descending once along the steepest gradient.
    //
    void descend(void)
    {
        itsTheta.copyTo(itsPriorTheta);
        itsTheta = itsTheta - (delta() * itsAlpha);
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
        cv::Mat result = cv::Mat::ones(theXs.rows, 3, CV_32F);
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

    // Compute a scaling factor for each dimension (column) of itsXs and
    // return it as a diagonal matrix after multiplying itsXs by the
    // result.
    //
    static cv::Mat scaleItsXs(cv::Mat &itsXs)
    {
        static const double epsilon = std::numeric_limits<float>::epsilon();
        cv::Mat result = cv::Mat::zeros(itsXs.cols, itsXs.cols, CV_32F);
        for (int i = 0; i < result.cols; ++i) {
            cv::Mat_<float> x = itsXs.col(i);
            double maximum, minimum;
            cv::minMaxLoc(x, &minimum, &maximum);
            const double difference = maximum - minimum;
            const double scale = difference > epsilon ? difference : 1.0;
            result.at<float>(i, i) = 1.0 / scale;
        }
        itsXs *= result;
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
        return itsTheta * itsScales;
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

    // Initialize gradient descent with learning rate alpha, feature
    // vectors theXs and result vectors theYs.  Start with theta [0,...].
    //
    GradientDescent(float alpha, const cv::Mat &theXs, const cv::Mat &theYs)
        : itsAlpha(alpha)
        , itsXs(makeItsXs(theXs))
        , itsYs(makeItsYs(theYs))
        , itsScales(scaleItsXs(itsXs))
        , itsPriorTheta(cv::Mat::zeros(1, itsXs.cols, CV_32F))
        , itsTheta(cv::Mat::zeros(1, itsXs.cols, CV_32F))
        , itsIterationCount(0)
    {
        DUMP(GradientDescent, itsYs.size());
        DUMP(GradientDescent, itsYs);
        DUMP(GradientDescent, itsXs.size());
        DUMP(GradientDescent, itsXs);
        DUMP(GradientDescent, itsTheta.size());
        DUMP(GradientDescent, itsTheta);
        DUMP(GradientDescent, itsScales.size());
        DUMP(GradientDescent, itsScales);
        DUMP(GradientDescent, itsScales.diag());
        DUMP(GradientDescent, itsScales.inv());
        cv::Mat unscaled = itsXs * itsScales.inv();
        DUMP(GradientDescent, unscaled);
        DUMP(GradientDescent, theXs);
        cv::Mat x = cv::Mat::ones(theXs.rows, 3, CV_32F);
        DUMP(GradientDescent, x.colRange(1, 3));
        theXs.copyTo(x.colRange(1, 3));
        // cv::hconcat(x, theXs, x);
        DUMP(GradientDescent, x);
        DUMP(GradientDescent, x - unscaled);
        DUMP(GradientDescent, cv::norm(x, unscaled));
        cv::Mat nz = x - unscaled;
        const double scaledNorm
            = cv::norm(x, unscaled) * cv::countNonZero(nz) / nz.total();
        DUMP(GradientDescent, scaledNorm);
    }
};


int main(int ac, const char *av[])
{
    static const cv::Vec3f zeros;
    DUMP(main, zeros);
    static int z[] = { 1, 2, 3, 4, 5, 6, 7, 8};
    static const size_t count = sizeof z / sizeof z[0];
    static const cv::Mat zints(2, 4, CV_32S, z);
    DUMP(main, zints);
    DUMP(main, zints.t());
    if (ac == 3) {
        std::ifstream xis(av[1]), yis(av[2]);
        float y; std::vector<float> yv; while (yis >> y) yv.push_back(y);
        cv::Mat_<float> theYs(yv);
        cv::Mat_<float> theXs(theYs.rows, 2, CV_32F);
        for (int i = 0; i < yv.size(); ++i) xis >> theXs(i, 0) >> theXs(i, 1);
        DUMP(main, theXs.rows);
        DUMP(main, theXs);
        static const float alpha = 0.07;
        GradientDescent gd(alpha, theXs, theYs);
        const cv::Mat theta1 = gd.theta(1);
        std::cerr << "theta1 == " << theta1 << std::endl;
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
