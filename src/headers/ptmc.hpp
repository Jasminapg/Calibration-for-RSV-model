#ifndef PTMC_HPP
#define PTMC_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <boost/random.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <vector>
#include <random>


using namespace Rcpp;
using namespace std;
using namespace Eigen;
using namespace boost::math;

std::random_device dev;
std::mt19937 engine(dev());
typedef boost::mt19937 PRNG_s;
PRNG_s rng(engine()); //Generate non-static random numbers (pick different numbers from prio distribution each run)

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins("cpp14")]]

// This section is taken from XXX and allows for quick samples from a multivaraite normal distribution. Depends on XX XX and XX.
namespace Eigen {
    
namespace internal {
    struct scalar_normal_dist_op
    {
      mutable boost::normal_distribution<double> norm;  // The gaussian combinator.

      EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

        template<typename Index>
        inline const double operator() (Index, Index = 0) const { return norm(rng); }
        inline void seed(const uint64_t &s) { rng.seed(s); }
    };

    template<>
    struct functor_traits<scalar_normal_dist_op >
    {
        enum { Cost = 50 * NumTraits<double>::MulCost, PacketAccess = false, IsRepeatable = false };
    };
} // end namespace internal


/**
Find the eigen-decomposition of the covariance matrix
and then store it for sampling from a multi-variate normal*/
// This is the function quickly samples from  multivariate normal (used in the proposal)
    class EigenMultivariateNormal
    {
      MatrixXd _covar;
      MatrixXd _transform;
      VectorXd _mean;
      //internal::scalar_normal_dist_op<double> randN; // Gaussian functor
      internal::scalar_normal_dist_op randN; // Gaussian functor
      bool _use_cholesky;
      SelfAdjointEigenSolver<MatrixXd > _eigenSolver; // drawback: this creates a useless eigenSolver when using Cholesky decomposition, but it yields access to eigenvalues and vectors

    public:
      EigenMultivariateNormal(const VectorXd& mean, const MatrixXd& covar,
                              bool use_cholesky=false, const uint64_t &seed=boost::mt19937::default_seed)
        :_use_cholesky(use_cholesky)
      {
        randN.seed(seed);
        setMean(mean);
        setCovar(covar);
      }

      void setMean(const VectorXd& mean) { _mean = mean; }
      void setCovar(const MatrixXd& covar)
      {
        _covar = covar;

        // Assuming that we'll be using this repeatedly,
        // compute the transformation matrix that will
        // be applied to unit-variance independent normals

        if (_use_cholesky)
        {
          Eigen::LLT<Eigen::MatrixXd > cholSolver(_covar);  //Decompose covar in the cholesky decomposition
          // We can only use the cholesky decomposition if
          // the covariance matrix is symmetric, pos-definite.
          // But a covariance matrix might be pos-semi-definite.
          // In that case, we'll go to an EigenSolver
          if (cholSolver.info()==Eigen::Success)
          {
            // Use cholesky solver
            _transform = cholSolver.matrixL(); //Retrieve matrix L from decomposition
          }
          else
          {
            _eigenSolver = SelfAdjointEigenSolver<MatrixXd >(_covar);  //computes Eigenvalues and vectors of self adjoint matrices
            _transform = _eigenSolver.eigenvectors()*_eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
          }
        }
        else
        {
          _eigenSolver = SelfAdjointEigenSolver<MatrixXd >(_covar);  //computes Eigenvalues and vectors of self adjoint matrices
          _transform = _eigenSolver.eigenvectors()*_eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal(); //find max then square root of eigen values x times vectors and take diganoal
        }
      }

      /// Draw nn samples from the gaussian and return them
      /// as columns in a Dynamic by nn matrix
      Matrix<double,Dynamic,-1> samples(int nn)
      {
        return (_transform * Matrix<double,Dynamic,-1>::NullaryExpr(_covar.rows(),nn,randN)).colwise() + _mean;
      }
    }; // end class EigenMultivariateNormal
} // end namespace Eigen


// Some useful functions
namespace ptmc {
    // Call a unform disitribution betweeen a and b
    double uniform_dist(double a, double b, char r, double x = 0)
    {
      if (r =='r')
      {boost::random::uniform_real_distribution<> u(a,b); return u(rng);}
      else if ( r =='m')
      {boost::math::uniform_distribution<> u(a,b); return mean(u);}
      else
      {boost::math::uniform_distribution<> u(a,b); return pdf(u,x);}
    }

    // Call a discriete unform disitribution a and b \in Z
    double uniform_dist_disc(int a, int b, char r, double x = 0)
    {
      if (r =='r')
      {boost::random::uniform_int_distribution<> u(a,b); return u(rng);}
      else
        return 0;
    }

    // Metropolis hasting Ratio
    double metropolisRatio(double LPN, double LPO, double temp)
    {
      double ratio;
      if(std::isnan(LPN - LPO))
        ratio = 0;
      else
        ratio = min(1.0, exp((LPN - LPO)/temp));

      return ratio;
    }

    // Randomly sample a proposal given the current (mu) a covariagne matrix, scale and the number of parameters P. s
    VectorXd proposal_sample(VectorXd curr, MatrixXd covar, double scale, int P)
    {
// if (scale > 1)
//       scale = 1;
      VectorXd trans_temp(P);

      EigenMultivariateNormal normX_solver(curr, scale*covar, true, engine());
      trans_temp = normX_solver.samples(1);

      return trans_temp;
    }
};

namespace ptmc{

template <typename D> // have no choice for parameter type
struct PTMC
{
  //! Keep track of means of the mcmc samples
  VectorXd acceptanceRate, lambda, Ms;
  MatrixXd c, covar_nA, covar_A;
  MatrixXd chain;
  MatrixXd current_mu, current;
  MatrixXd chainLP;
  VectorXd currentLP;
  int iterations, run_s, thin, burn, adap_Cov_burn, con_up, M, P, adap_Cov_freq, adap_Temp_freq;
  bool debug, adap_Cov, adap_Temp;

  List settings;

  std::function<VectorXd()> gen_init;
  std::function<double(VectorXd)> eval_lpr;
  std::function<double(D,VectorXd)> eval_ll;

  PTMC() {}

  double get_lp(D data, VectorXd param)
  {
      double lpr = this->eval_lpr(param);
        if (isinf(lpr))
            return log(0);
      
      double ll = this->eval_ll(data, param);
      return lpr + ll;
  }

  ~PTMC() {}


  // Initialise the MCMCPT structure
  void initialise(PTMC<D>& PTMC_t, D data, List settings){

    PTMC_t.M = settings["M"];
    PTMC_t.P = settings["P"];
    PTMC_t.iterations = settings["iterations"];
    PTMC_t.thin = settings["thin"];
    PTMC_t.burn = settings["burn"];
    PTMC_t.adap_Cov_burn = settings["adap_Cov_burn"];
    PTMC_t.con_up = settings["consoleUpdates"];
    PTMC_t.adap_Cov = settings["adap_Cov"];
    PTMC_t.adap_Temp = settings["adap_Temp"];
    PTMC_t.adap_Cov_freq = settings["adap_Cov_freq"];
    PTMC_t.adap_Temp_freq = settings["adap_Temp_freq"];
    PTMC_t.debug = settings["Debug"];

    PTMC_t.run_s = (PTMC_t.iterations-PTMC_t.burn)/(PTMC_t.thin);
    PTMC_t.chain = MatrixXd::Zero(PTMC_t.M*PTMC_t.run_s,PTMC_t.P+2);

    VectorXd init;
    PTMC_t.currentLP = VectorXd::Zero(PTMC_t.M);
    PTMC_t.current_mu = MatrixXd::Zero(PTMC_t.M,PTMC_t.P);
    PTMC_t.current = MatrixXd::Zero(PTMC_t.M,PTMC_t.P);
    double ll;

    for (int m = 0; m < PTMC_t.M; m++)
    {
      init = PTMC_t.gen_init();
      ll = PTMC_t.get_lp(data, init);
      while(isinf(ll) || isnan(ll))
      {
        init = PTMC_t.gen_init();
        ll = PTMC_t.get_lp(data, init);
      }

      PTMC_t.current.row(m) = init;
      PTMC_t.current_mu.row(m) = init;

      PTMC_t.currentLP(m) = ll;
    }


    PTMC_t.covar_nA = MatrixXd::Zero(PTMC_t.P ,PTMC_t.P );
    PTMC_t.covar_A = MatrixXd::Zero(PTMC_t.M*PTMC_t.P ,PTMC_t.P );
    for(int i = 0; i < PTMC_t.P ; i++)
    {
      PTMC_t.covar_nA(i,i) = 1.0;
    }
    for (int m = 0; m < PTMC_t.M; m++)
    {
      for(int i = 0; i < PTMC_t.P ; i++)
        PTMC_t.covar_A(m*PTMC_t.P+i,i) = 1.0;
    }

    PTMC_t.lambda = VectorXd::Zero(PTMC_t.M);
    PTMC_t.Ms = VectorXd::Zero(PTMC_t.M);
    for(int i = 0; i < PTMC_t.M; i++)
    {
        PTMC_t.lambda(i) = log(0.1*0.1/(double)PTMC_t.P);
        PTMC_t.Ms(i) = log(2.382*2.382/(double)PTMC_t.P);
    }
  }


  // function to get the posterior sample
  MatrixXd run_ptmc_C(PTMC<D>& PTMC_t, D data, List settings)
  {
    initialise(PTMC_t, data, settings);
      
    int burn = PTMC_t.burn; // burn int
    int thin = PTMC_t.thin; // thining
    int M = PTMC_t.M;       // Number chains in temperature ladder
    int m_step = M/2;

    int P = PTMC_t.P;       // Number of parameters
    int adap_Cov_burn = PTMC_t.adap_Cov_burn;// Number of steps to run before the adaptive covariance matrix starts
    int l = PTMC_t.run_s;       // Number of samples in the posterior
    int iterations = PTMC_t.iterations; // Number of iterations
    bool debug = PTMC_t.debug;        // Wether to drug a debug
    bool adap_Cov = PTMC_t.adap_Cov;  // Whether to include the adaptive covariacne matrix
    bool adap_Temp = PTMC_t.adap_Temp;  // Whether to include an adaptive temperature ladder
    int adap_Cov_freq = PTMC_t.adap_Cov_freq;  // Whether to include the adaptive covariacne matrix
    int adap_Temp_freq = PTMC_t.adap_Temp_freq;  // Whether to include an adaptive temperature ladder

    vector<int> counterFunEval(M, 0);
    vector<int> counterAccept(M, 0);
    vector<int> counter(M, 0);
    vector<int> counter_adapt(M, 0);
    vector<int> counter_nonadapt(M, 0);
    vector<double> acceptanceRate(M, 0);

    vector<int> counterFunEvalTemp(M, 0);
    vector<int> counterAcceptTemp(M, 0);
    vector<double> tempering, S;

    MatrixXd cov, chain_m;
    VectorXd chain_m_i, swap1, swap2;
    VectorXd proposal;
    double swap1D, swap2D;
    double proposalLP;
    double alpha;
    bool accepted, adaptive;
    double gf;

    for (int i = 0; i < M; i++)
      tempering.push_back(pow(10, 7.0*(i)/(M-1.0)));

    for (int i = 0; i < M-1; i++)
      S.push_back(log(log(tempering[i+1])-log(tempering[i])));

    for (int i = 0; i < iterations; i++){
      for (int m = 0; m < M; m++){
        if (debug) Rcout << "METROPOLIS SECTION. Iteration number: " << i << ". Temperature ladder number: " << m << ". ";
          accepted = false;
          if (i < adap_Cov_burn || uniform_dist(0, 1,'r') < 0.05 || !adap_Cov || m > m_step){
            counter_nonadapt[m]++; adaptive = false;
            cov = PTMC_t.covar_nA;
            proposal = proposal_sample(PTMC_t.current.row(m), cov, exp(PTMC_t.lambda[m]), P);
            proposalLP = PTMC_t.get_lp(data, proposal);
            alpha = metropolisRatio(proposalLP, PTMC_t.currentLP(m), tempering[m]);
          }
          else{
            counter_adapt[m]++; adaptive = true;
            cov = PTMC_t.covar_A.block(m*P,0,P,P);
            proposal = proposal_sample(PTMC_t.current.row(m), cov, exp(PTMC_t.Ms[m]), P);
            proposalLP = PTMC_t.get_lp(data, proposal);
            alpha = metropolisRatio(proposalLP, PTMC_t.currentLP(m), tempering[m]);
          }
          counterFunEval[m]++;
          
          // Update the parameters
          if (adaptive){
            PTMC_t.Ms[m] += pow(1+counter_adapt[m],-0.5)*(alpha - 0.234);
              if (isinf(PTMC_t.Ms[m]) || isnan(PTMC_t.Ms[m])){
                  stop("Ms is infinite or not a number.");
              }
          }
          else{
            PTMC_t.lambda[m] += pow(1+counter_nonadapt[m],-0.5)*(alpha - 0.234);
              if (isinf(PTMC_t.lambda[m]) || isnan(PTMC_t.lambda[m])){
                  stop("lambda is infinite or not a number.");
              }
          }
          // If accepted
          if (uniform_dist(0, 1,'r') < alpha ){
            accepted = true; counterAccept[m]++;
            PTMC_t.current.row(m) = proposal;
            PTMC_t.currentLP(m) = proposalLP;
          }

          if((i > (burn-1)) && (i%thin == 0) ){

            for (int p = 0; p < P; p++){
              PTMC_t.chain(m*l+counter[m], p) = PTMC_t.current(m,p);
            }
            PTMC_t.chain(m*l+counter[m], P) = PTMC_t.currentLP(m);
            PTMC_t.chain(m*l+counter[m], P+1) = tempering[m];

            counter[m]++;
          }

          if(i%adap_Cov_freq == 0 && i > adap_Cov_burn){
            int iA = i-adap_Cov_burn;  gf = pow(1+iA,-0.5);
            if (iA == 1){
              PTMC_t.current_mu.row(m) = PTMC_t.current.row(m);
              PTMC_t.Ms[m] = PTMC_t.Ms[m]*PTMC_t.lambda[m];
            }
            else
            {
              PTMC_t.current_mu.row(m) = PTMC_t.current_mu.row(m) + gf*(PTMC_t.current.row(m)-PTMC_t.current_mu.row(m));
              PTMC_t.covar_A.block(m*P,0,P,P) = PTMC_t.covar_A.block(m*P,0,P,P) + gf*((PTMC_t.current.row(m)-PTMC_t.current_mu.row(m)).transpose()*((PTMC_t.current.row(m))-(PTMC_t.current_mu.row(m)))) - gf*PTMC_t.covar_A.block(m*P,0,P,P);
            }
          }
          if (debug) Rcout << "End of iteration: " << m << "." << "\r";
      }

      if(i%adap_Temp_freq == 0)
      {
        for (int m = 0; m < M; m++){
          int p = uniform_dist_disc(0, M - 2, 'r');
          counterFunEvalTemp[p] ++;
          int r = metropolisRatio(PTMC_t.currentLP(p+1), PTMC_t.currentLP(p), tempering[p]*tempering[p+1]/(tempering[p+1]-tempering[p]));
          if (uniform_dist(0, 1, 'r') < r)
          {
            counterAcceptTemp[p]++;
            swap1D = PTMC_t.currentLP(p); swap2D = PTMC_t.currentLP(p+1);
            PTMC_t.currentLP(p) = swap2D; PTMC_t.currentLP(p+1) = swap1D;

            swap1 = PTMC_t.current.row(p); swap2 = PTMC_t.current.row(p+1);
            PTMC_t.current.row(p) = swap2; PTMC_t.current.row(p+1) = swap1;
          }

            S[p] += pow((1+counterFunEvalTemp[p]),(-0.5))*(r - 0.234);
        }

        if (adap_Temp){
          for (int m = 0; m < M-1; m++){
            if (exp(S[m]) < 0 || isinf(exp(S[m]))||isnan(exp(S[m])))
              stop("exp(S[m]) is either negative or infinite/nan. Value: ", exp(S[m]));
            double expS;
            if (S[m] < -1){
              expS = exp(-1);
            }
            else{
              expS = exp(S[m]);
            }
              
            tempering[m+1] = tempering[m]*exp(expS);

              if (tempering[m] < 0 || isinf(tempering[m])||isnan(tempering[m]))
                stop("tempering[m] is either negative or infinite/nan. Value: ", tempering[m]);
          }
        }
      }
      if(i%PTMC_t.con_up == 0 && !debug)
        Rcout << "Running MCMC-PT iteration number: " << i << " of " << iterations << ". Current logpost: " << PTMC_t.currentLP(0) << ". " << PTMC_t.currentLP(1) << "           " << "\r";
    }

    return PTMC_t.chain.block(0, 0, PTMC_t.run_s, PTMC_t.P+2);
  }

};

};
// namespace ptmc
#endif

