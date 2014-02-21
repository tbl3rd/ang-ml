#include <opencv2/core.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>


// Do this exercise.
//
// http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html

#define DUMP(L, X) std::cerr << #L ": " #X " == " << X << std::endl;


static void showUsage(const char *av0)
{
    std::cout << av0 << ": Demonstrate gradient descent." << std::endl
              << std::endl
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
    const cv::Mat itsXs;
    const cv::Mat itsY;
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

    // Return a m*2 matrix of X vectors with X[0] == 1.
    //
    static cv::Mat makeItsX(const std::vector<float> &xv)
    {
        cv::Mat_<float> result = cv::Mat::ones(xv.size(), 2, CV_32F);
        for (int i = 0; i < xv.size(); ++i) result(i, 1) = xv[i];
        return result;
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

public:

    // Return the theta resulting from n descents.
    //
    const cv::Mat &theta(int n)
    {
        if (n > itsIterationCount) {
            size_t rest = n - itsIterationCount;
            while(rest--) descend();
        }
        return itsTheta;
    }

    // Return the Theta vector after meeting tc.
    //
    const cv::Mat &operator()(const cv::TermCriteria &tc)
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
    // vectors xv and result vectors yv.  Start with theta [0,...].
    //
    GradientDescent(float alpha,
                    const std::vector<float> &xv,
                    const std::vector<float> &yv)
        : itsAlpha(alpha)
        , itsX(makeItsX(xv))
        , itsY(yv)
        , itsPriorTheta(cv::Mat::zeros(2, 1, CV_32F))
        , itsTheta(cv::Mat::zeros(2, 1, CV_32F))
        , itsIterationCount(0)
    {
        // DUMP(GradientDescent, itsY.size());
        // DUMP(GradientDescent, itsX.size());
        // DUMP(GradientDescent, itsTheta.size());
    }
};


// Tabulate the data on os from xv, yv, and theta.
//
#define SHOW(X) std::fixed << std::setprecision(4) << std::setw(10) << X
static void showData(std::ostream &os,
                     const std::vector<float> &xv,
                     const std::vector<float> &yv,
                     const cv::Mat &theta)
{
    const float theta0 = theta.at<float>(0);
    const float theta1 = theta.at<float>(1);
    os << "y = " << std::fixed << std::setprecision(4) << theta0
       << " + "  << std::fixed << std::setprecision(4) << theta1
       << " * x" << std::endl << std::endl;
    os << std::setw(3) << "n" << SHOW("X[n]") << SHOW("Y[n]")
       << SHOW("H(X[n])") << SHOW("Error") << std::endl;
    os << std::setw(3) << "-" << SHOW("----") << SHOW("----")
       << SHOW("-------") << SHOW("-----") << std::endl;
    for (int i = 0; i < xv.size(); ++i) {
        const float x = xv[i];
        const float y = yv[i];
        const float z = theta0 + theta1 * x;
        const float d = std::abs(y - z);
        os << std::setw(3) << i
           << SHOW(x) << SHOW(y) << SHOW(z) << SHOW(d) << std::endl;
    }
}
#undef SHOW


// Report on os 1500 iterations of gradient descent on alpha, xv, and yv.
//
static void run1500iterations(std::ostream &os, float alpha,
                              const std::vector<float> &xv,
                              const std::vector<float> &yv)
{
    GradientDescent gd(alpha, xv, yv);
    const cv::Mat theta1 = gd.theta(1);
    os << std::endl << "theta1    == " << theta1 << std::endl;
    const cv::Mat theta1500 = gd.theta(1500);
    os << "theta1500 == " << theta1500 << std::endl << std::endl;
    showData(os, xv, yv, theta1500);
}


// Report on os gradient descent on alpha, xv, and yv until some
// termination criteria are met.
//
static void withTermCriteria(std::ostream &os, float alpha,
                             const std::vector<float> &xv,
                             const std::vector<float> &yv)
{
    static const int criteria
        = cv::TermCriteria::COUNT | cv::TermCriteria::EPS;
    static const float epsilon = 0.0000001;
    static const int iterations = 1500;
    static const cv::TermCriteria tc(criteria, iterations, epsilon);
    os << std::endl << "Now use TermCriteria: "
       << tc.maxCount << " iterations or an epsilon of "
       << std::scientific << tc.epsilon
       << std::endl;
    GradientDescent gdTc(alpha, xv, yv);
    const cv::Mat theta = gdTc(tc);
    os << "TermCriteria met after " << gdTc.count() << " iterations."
       << std::endl << std::endl;
    showData(os, xv, yv, theta);
}


int main(int ac, const char *av[])
{
    if (ac == 3) {
        std::vector<float> xv, yv;
        std::ifstream xis(av[1]), yis(av[2]);
        float z;
        while (xis >> z) xv.push_back(z);
        while (yis >> z) yv.push_back(z);
        static const float alpha = 0.07;
        run1500iterations(std::cout, alpha, xv, yv);
        withTermCriteria(std::cout, alpha, xv, yv);
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
