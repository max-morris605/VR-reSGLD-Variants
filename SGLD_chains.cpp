#include <iostream>
#include <vector>
#include <numeric>       // std::iota
#include <random>
#include <chrono>
#include <limits>
#include <Eigen/Dense>
#include <fstream>


using namespace Eigen ;

const double MY_PI = 3.141592653589793;


double energy(double *x, double beta) 
{
    double d1 = *x - beta ;

    double d2 = *x - (20.0 - beta) ;

    double a  = std::exp(-d1 * d1 / 50.0) ;
     
    double b  = std::exp(-d2 * d2 / 50.0) ;

    return - std::log( a + b ) ;

}

double energy_derivative(Eigen::VectorXd& data, double beta) {

    double sum = 0.0 ;

    for (int i = 0; i < data.size(); i++)
    {
        double d1 = data(i) - beta ;

        double d2 = data(i) - 20.0 + beta ;

        double a  = std::exp(-(d1 * d1) / 50.0) ;
        
        double b  = std::exp(-(d2 * d2) / 50.0) ;

        double num = d2 * b - d1 *a ;

        double denom = a + b ;

        sum += (num / 25.0) / denom ;

    }

    return sum ;
}

int main() {

    std::mt19937_64 rng(std::random_device{}());

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    std::normal_distribution<double> normal01(0.0, 1.0);


    int N = 100000 ; 
           
    int n = 5000 ;     
         
    int k_max  = 10000 ;  
           
    double eta = 0.0002;           

    double temp1 = 10.0 ;

    double temp2 = 2500.0 ;

    double ratio  = static_cast<double>(N) / n ;

    double coeff1 = std::sqrt(2.0 * temp1 * eta) ;

    double coeff2 = std::sqrt(2.0 * temp2 * eta) ;

    Eigen::VectorXd train_data(N) ;

    for (int i = 0; i < N; ++i) 
    {
        double u = uniform01(rng) ;

        double z = normal01(rng) ;

            if (u < 0.5) 
        {
            train_data(i) = 5 * z - 5 ;     
        } 
        else 
        {
            train_data(i) = 5 * z + 25 ;    
        }
    }

    double beta1 = 0.0 ;

    double beta2 = 0.0;

    double grad1 ;;

    double grad2 ;

    double xi1 ;

    double xi2 ;

    Eigen::VectorXd chain1(k_max), chain2(k_max) ;

    chain1(0) = beta1 ;

    chain2(0) = beta2 ;

    std::vector<int> indices(N) ;

    std::iota(indices.begin(), indices.end(), 0) ;

    for (int k = 1; k < k_max; ++k) {

        std::shuffle(indices.begin(), indices.end(), rng) ;

        VectorXd sample_set(n) ;

        for (int i = 0; i < n; ++i) {
            sample_set(i) = train_data(indices[i]);
        }

        grad1 = energy_derivative(sample_set, beta1) ;

        grad2 = energy_derivative(sample_set, beta2) ;

        xi1 = normal01(rng) ;

        xi2 = normal01(rng) ;

        beta1 += -eta * ratio * grad1 + coeff1 * xi1 ;

        beta2 += -eta * ratio * grad2 + coeff2 * xi2 ;

        chain1(k) = beta1 ;

        chain2(k) = beta2 ;
    }

    std::ofstream myfile1 ;

    myfile1.open("chain1.txt") ;

    myfile1 << chain1 << '\n' ;

    std::ofstream myfile2 ;

    myfile2.open("chain2.txt") ;

    myfile2 << chain2 << '\n' ;

    //std::cout << chain1.transpose() << std::endl ;

    //std::cout << chain2.transpose() << std::endl;

    return 0;
}
