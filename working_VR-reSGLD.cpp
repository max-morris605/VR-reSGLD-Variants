#include <iostream>
#include <vector>
#include <numeric>       // std::iota
#include <random>
#include <chrono>
#include <limits>
#include <Eigen/Dense>
#include <fstream>

const double MY_PI = 3.141592653589793;

void indice_sampler(std::vector<int>& indices, std::mt19937_64& rng) 
{
    std::shuffle(indices.begin(), indices.end(), rng);
}



double energy(double* x, double beta) 
{
    double d1 = *x - beta ;

    double d2 = *x - 20.0 + beta ;

    double a  = std::exp(-(d1 * d1) / 50.0) ;
     
    double b  = std::exp(-(d2 * d2) / 50.0) ;

    return - std::log( a + b ) ;

}

double global_update(Eigen::VectorXd& pTrain_Data, double beta)
{
    double sum = 0.0 ;

    double N = pTrain_Data.size() ;

    for(int i = 0; i < N; i++)
    {
        sum += energy(&pTrain_Data(i), beta) ;
    }

    return sum ;
}

double energy_derivative(double* x, double beta) {

    double d1 = *x - beta ;

    double d2 = *x - 20.0 + beta ;

    double a  = std::exp(-(d1 * d1) / 50.0) ;
     
    double b  = std::exp(-(d2 * d2) / 50.0) ;

    double num = d2 * b - d1 *a ;

    double denom = a + b ;

    return (num / 25.0) / denom ;
}

double swap_rate(double sum1, double sum2, double temp1, double temp2, int N, int n, double sigma, double F)
{
    double ratio ;

    ratio = static_cast<double>(N) / n ;

    double temp_diff ;

    temp_diff = (1 / temp1) - (1 / temp2) ;

    double energy_diff = sum1 - sum2 ;

    double bias_term = temp_diff * sigma / F ;

    double exponent = temp_diff * (energy_diff - bias_term) ;

    //min(S,1)

    if (exponent > 0.0) 
    {
        exponent = 0.0 ;
    }

    return std::exp(exponent) ;
}

int main() {

    using clock = std::chrono::steady_clock ; // monotonic, no jumps

    auto start = clock::now() ;

    std::mt19937_64 rng(std::random_device{}());

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    std::normal_distribution<double> normal01(0.0, 1.0);


    int N = 1000 ; 
           
    int n = 100;     
         
    int k_max  = 1000000 ;  
           
    double eta = 0.002 ;           

    double temp1 = 50.0 ;

    double temp2 = 750.0 ;

    double smooth_factor = 0.1 ;

    double correct_factor = 1 ;

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

    double beta2 = 0.0 ;

    double grad1 ;;

    double grad2 ;

    double xi1 ;

    double xi2 ;

    double energy_sum1 ;

    double energy_sum2 ;

    Eigen::VectorXd chain1(k_max), chain2(k_max), swap_rate_lst(k_max - 1), sigma_lst(k_max) ;

    chain1(0) = beta1 ;

    chain2(0) = beta2 ;

    std::vector<int> indices(N) ;

    std::iota(indices.begin(), indices.end(), 0) ;

    int var_sample_number = 20 ;

    //initialise sigma hat


    double init_mean = 0.0 ;

    double init_sum_square = 0.0 ;


    for (int idx = 0; idx < var_sample_number; idx++)
    {
        indice_sampler(indices, rng) ;

        Eigen::VectorXd subsample = train_data(Eigen::all, indices).col(0).head(n) ;

        double sum = 0.0 ;

        for(int j = 0; j < n; j++)
        {
            sum += energy(&subsample(j), beta1) ;
        }

        double x = ratio * sum ;

        init_mean += x / var_sample_number ;

        init_sum_square += x * x ;
        
    }

    double sigma_hat = (init_sum_square - var_sample_number * init_mean * init_mean) / (var_sample_number - 1) ;

    sigma_lst(0) = sigma_hat ;

    double beta1_VR = beta1 ;

    double beta2_VR = beta2 ;

    int swaps = 0 ;

    double global_sum1 = global_update(train_data, beta1_VR) ;

    double global_sum2 = global_update(train_data, beta2_VR) ;



    for (int k = 1; k < k_max; ++k) {

        

        if(k % 200 == 0)
        {

            double mean1 = 0.0 ;

            double mean2 = 0.0 ;

            double square_sum1 = 0.0;

            double square_sum2 = 0.0;

            for (int idx = 0; idx < var_sample_number; idx++)
            {
                indice_sampler(indices, rng) ;

                Eigen::VectorXd subsample = train_data(Eigen::all, indices).col(0).head(n) ;

                double sum1 = global_sum1 ;

                double sum2 = global_sum2 ;

                for(int j = 0; j < n; j++)
                {
                    sum1 += energy(&subsample(j), beta1) - energy(&subsample(j), beta1_VR);

                    sum2 += energy(&subsample(j), beta2) - energy(&subsample(j), beta2_VR) ;
                }

                double x1 = ratio * sum1 ;

                double x2 = ratio * sum2 ;

                mean1 += x1 / var_sample_number ;

                mean2 += x2 / var_sample_number ;

                square_sum1 += x1 * x1 ;

                square_sum2 += x2 * x2 ;
            }

            double sigma1 = (square_sum1 - var_sample_number * mean1 * mean1) / (var_sample_number - 1) ;

            double sigma2 = (square_sum2 - var_sample_number * mean2 * mean2) / (var_sample_number - 1) ;

            sigma_hat = (1 - smooth_factor) * sigma_hat + smooth_factor * (sigma1 + sigma2) / 2 ;

            beta1_VR = beta1 ;

            beta2_VR = beta2 ;

            global_sum1 = global_update(train_data, beta1_VR) ;

            global_sum2 = global_update(train_data, beta2_VR) ;

        }

        indice_sampler(indices, rng) ;

        Eigen::VectorXd sample_set = train_data(Eigen::all, indices).col(0).head(n);

        grad1 = 0.0 ;

        grad2 = 0.0 ;

        Eigen::VectorXd& pSample_Set = sample_set ;

        for (int j = 0; j < n; ++j) 
        {
            grad1 += energy_derivative(&pSample_Set(j), beta1) ;

            grad2 += energy_derivative(&pSample_Set(j), beta2) ;
        }

        xi1 = normal01(rng) ;

        xi2 = normal01(rng) ;

        beta1 += -eta * ratio * grad1 + coeff1 * xi1 ;

        beta2 += -eta * ratio * grad2 + coeff2 * xi2 ;

        //

        energy_sum1 = global_sum1 ;

        energy_sum2 = global_sum2 ;

        for (int b = 0; b < n; b++)
        {
            energy_sum1 += ratio * ( energy(&pSample_Set(b), beta1) - energy(&pSample_Set(b), beta1_VR) ) ;

            energy_sum2 += ratio * ( energy(&pSample_Set(b), beta2) - energy(&pSample_Set(b), beta2_VR) ) ;
        }

        double S = swap_rate(energy_sum1, energy_sum2, temp1, temp2, N, n, sigma_hat, correct_factor) ;

        swap_rate_lst(k-1) = S ;

        double u = uniform01(rng) ;

        if (u < S){

            double tmp = beta1 ;

            beta1 = beta2 ;

            beta2 = tmp ;

            swaps += 1 ;
        }

        sigma_lst(k) = sigma_hat ;

        chain1(k) = beta1 ;

        chain2(k) = beta2 ;
    }

    auto end = clock::now();

    std::ofstream(myfile) ;

    std::ofstream(swap) ;

    swap.open("swap_rate_lst.txt") ;

    swap << swap_rate_lst << '\n' ;

    swap.close() ;

    myfile.open("VR-reSGLD_chain.txt") ;

    myfile << chain1 << '\n' ;

    myfile.close() ;

    std::cout << swaps << '\n' ; 
    
    std::chrono::duration<double> elapsed = end - start ; 

    std::cout << "Elapsed: " << elapsed.count() << " s\n" ;

    return 0 ;

}
