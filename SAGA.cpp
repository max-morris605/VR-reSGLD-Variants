#include <iostream>
#include <vector>
#include <numeric>       // std::iota
#include <random>
#include <chrono>
#include <limits>
#include <Eigen/Dense>
#include <fstream>
#include <algorithm>

const double MY_PI = 3.141592653589793;

void indice_sampler(std::vector<int>& indices, std::mt19937_64& rng)
{
    std::shuffle(indices.begin(), indices.end(), rng) ;
}


double energy(double* x, double beta) 
{
    double d1 = *x - beta ;

    double d2 = *x - 20.0 + beta ;

    double a  = std::exp(-(d1 * d1) / 50.0) ;
     
    double b  = std::exp(-(d2 * d2) / 50.0) ;

    return - std::log( a + b ) ;

}

double energy_derivative_scalar(double* x, double beta) 
{
     double d1 = *x - beta;
     double d2 = *x - 20.0 + beta;
     double a  = std::exp(-(d1 * d1) / 50.0);
     double b  = std::exp(-(d2 * d2) / 50.0);
     double num = d2 * b - d1 * a;
     double denom = a + b;
     return (num / 25.0) / denom;
}


double swap_rate(double sum1, double sum2, double temp1, double temp2,
                 int /*N*/, int /*n*/, double sigma_var, double F)
{
    double temp_diff = (1.0 / temp1) - (1.0 / temp2) ;

    double energy_diff = sum1 - sum2 ;         

    double bias_term  = (temp_diff) * (sigma_var / F) ; 

    double exponent = temp_diff * (energy_diff - bias_term);

    // min(S,1) 

    if (exponent > 0.0) 
    {
        exponent = 0.0 ;
    }

    return std::exp(exponent) ;
}

int main() {

    using clock = std::chrono::steady_clock ; 

    auto start = clock::now() ;

    std::mt19937_64 rng(std::random_device{}());

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    std::normal_distribution<double> normal01(0.0, 1.0);


    const int N = 1000 ; 
           
    const int n = 200;     
         
    const int k_max  = 100000 ;  
           
    const double eta = 0.02 ;           

    const double temp1 = 25.0 ;

    const double temp2 = 1000.0 ;

    const double smooth_factor = 0.2 ;

    const double correct_factor = 1 ;

    const double noise1 = std::sqrt(2.0 * temp1 * eta) ;

    const double noise2 = std::sqrt(2.0 * temp2 * eta) ;

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

    Eigen::VectorXd chain1(k_max), chain2(k_max), swap_rate_lst(k_max - 1) ;

    chain1(0) = beta1 ;

    chain2(0) = beta2 ;

    //SAGA Initialization

    Eigen::VectorXd stored_SAGA_energies1(N), stored_SAGA_energies2(N) ;

    for(int idx = 0; idx < N; idx++)
    {
        stored_SAGA_energies1(idx) = energy(&train_data(idx), beta1) ;

        stored_SAGA_energies2(idx) = energy(&train_data(idx), beta2) ;
    }

    double SAGA_mean1, SAGA_mean2 ;

    SAGA_mean1 = stored_SAGA_energies1.sum() / double(N) ;

    SAGA_mean2 = stored_SAGA_energies2.sum() / double(N) ;

    //EMA initialization

    double mu = 0.0, sigma_hat = 0.0 ;

    std::vector<int> indices(N) ;

    std::iota(indices.begin(), indices.end(), 0) ;

    const int SAGA_sample_size = 5 ;

    Eigen::VectorXd new_SAGA_energies1(SAGA_sample_size), 
    new_SAGA_energies2(SAGA_sample_size), 
    SAGA_deltas1(SAGA_sample_size), 
    SAGA_deltas2(SAGA_sample_size) ;
    
    
    
    std::vector<int> idx(n), saga_idx(SAGA_sample_size);             

 	std::uniform_int_distribution<int> uid(0, N - 1);

    int swaps = 0 ;

    


    for (int k = 1; k < k_max; ++k) 
    {
        indice_sampler(indices, rng);

 	    std::copy_n(indices.begin(), n, idx.begin());

        // --- gradients (already fixed) ---
        grad1 = 0.0 ;

        grad2 = 0.0 ;

        for (int j = 0; j < n; j++)
        {
            grad1 += energy_derivative_scalar(&train_data(idx[j]), beta1) ;

            grad2 += energy_derivative_scalar(&train_data(idx[j]), beta2);
        }

        // Langevin step
        xi1 = normal01(rng) ; 
        
        xi2 = normal01(rng) ;

        beta1 += -eta * (double(N)/n) * grad1 + noise1 * xi1 ;

        beta2 += -eta * (double(N)/n) * grad2 + noise2 * xi2 ;

        // --- compute SAGA deltas using the *old* storage snapshot ---
        double curr_sum1 = 0.0 ;
    
        double curr_sum2 = 0.0 ;

        indice_sampler(indices, rng);

 	    std::copy_n(indices.begin(), SAGA_sample_size, saga_idx.begin());


        for (int j = 0; j < SAGA_sample_size; ++j) 
        {
            new_SAGA_energies1(j) = energy(&train_data(saga_idx[j]), beta1) ;

            new_SAGA_energies2(j) = energy(&train_data(saga_idx[j]), beta2) ;

            SAGA_deltas1(j) = new_SAGA_energies1(j) - stored_SAGA_energies1(saga_idx[j]) ;

            SAGA_deltas2(j) = new_SAGA_energies2(j) - stored_SAGA_energies2(saga_idx[j]) ;

            curr_sum1 += SAGA_deltas1(j) ;

            curr_sum2 += SAGA_deltas2(j) ;
        }

        // --- form energy sums (still with old means!) ---
        energy_sum1 = N * (curr_sum1 / SAGA_sample_size + SAGA_mean1) ;

        energy_sum2 = N * (curr_sum2 / SAGA_sample_size + SAGA_mean2) ;

        // --- update variance from Δ ---
        
        
        
        
    
        // --- 2) VARIANCE-ONLY batches (fast: sample with replacement, no shuffles, no Eigen allocs) ---
        const int K = 10;  // you can start with 10–20, then reduce to 5 later
        std::uniform_int_distribution<int> uid(0, N - 1);

        double dsum = 0.0, d2sum = 0.0;

        for (int krep = 0; krep < K; ++krep) {
            double s1 = 0.0, s2 = 0.0;
            // sample-with-replacement minibatches of size n for each chain
            for (int j = 0; j < n; ++j) {
                const int id = uid(rng);

                const double e1 = energy(&train_data(id), beta1);
                const double e2 = energy(&train_data(id), beta2);

                s1 += (e1 - stored_SAGA_energies1(id));
                s2 += (e2 - stored_SAGA_energies2(id));
            }
            const double L1 = N * (s1 / n + SAGA_mean1);
            const double L2 = N * (s2 / n + SAGA_mean2);
            const double d  = L1 - L2;

            dsum  += d;
            d2sum += d * d;
        }

        const double dbar     = dsum / K;
        const double var_step = (K > 1) ? (d2sum - K * dbar * dbar) / (K - 1) : 0.0;

        // EMA for variance (keep it gentle; you set 0.2 which is quite high)
        sigma_hat = (1 - smooth_factor) * sigma_hat + smooth_factor * std::max(0.0, var_step);
    

        
        // --- NOW update SAGA storage and means (apply deltas once) ---
        double delta_mean1 = 0.0 ;

        double delta_mean2 = 0.0 ;

        for (int j = 0; j < SAGA_sample_size; ++j) 
        {
            delta_mean1 += (new_SAGA_energies1(j) - stored_SAGA_energies1(saga_idx[j])) ;

            delta_mean2 += (new_SAGA_energies2(j) - stored_SAGA_energies2(saga_idx[j])) ;

            stored_SAGA_energies1(saga_idx[j]) = new_SAGA_energies1(j) ;

            stored_SAGA_energies2(saga_idx[j]) = new_SAGA_energies2(j) ;
        }

        SAGA_mean1 += delta_mean1 / double(N) ;

        SAGA_mean2 += delta_mean2 / double(N) ;

        // --- swap decision (storage now remains attached to temperatures) ---
        double S = swap_rate(energy_sum1, energy_sum2, temp1, temp2, N, n, sigma_hat, correct_factor) ;

        swap_rate_lst(k-1) = S ;

        if (uniform01(rng) < S) 
        {
            std::swap(beta1, beta2) ; 
            
            ++swaps ;
        }

        chain1(k) = beta1 ;

        chain2(k) = beta2 ;

    }

    auto end = clock::now();

    std::ofstream(myfile) ;

    std::ofstream(swap) ;

    swap.open("swap_rate_lst.txt") ;

    swap << swap_rate_lst << '\n' ;

    swap.close() ;

    myfile.open("SAGA-VR-reSGLD_chain.txt") ;

    myfile << chain1 << '\n' ;

    myfile.close() ;

    std::cout << swaps << '\n';

    std::chrono::duration<double> elapsed = end - start ; 

    std::cout << "Elapsed: " << elapsed.count() << " s\n" ;


    return 0 ;

}
