# `ptmc` R Package 

## Parallel Tempering Markov Chain Monte Carlo 


###Overview

This repository contains an R Package which takes an arbitrary likelihood and prior function for a set of fitted parameters and samples posterior values via a Metropolis Hastings algorithm with Parallel Tempering. This implementation is a generalised version of the algorithm outlined by [Miasojedow et al. 2012](https://arxiv.org/pdf/1205.1076.pdf).

###Brief overview of Parallel Tempering MCMC

Parallel Tempering runs $M$ chains simulationeously, each of which are updated via the Metropolis-Hastings algorithm, where the metropolis ratio for a  Target Distribution $\pi$ of a Markov chain $\theta_i^m$ for chain $m$ at iteration $i$ is given by 

$$\alpha(\theta_i^m, \theta^*) = \left(\frac{\pi (\theta^*)}{\pi (\theta^m_i)}\right)^{1/T^m}$$

where $\theta^*$ is a proposed position, and $T^m$ is such that $T^0$ = 1, and $T^{m+1} > T^m$ (monotonicity). When $T^m$ is small (cold) the chains run similar to a random-walk Metropolis Hasting algorihtm. However then $T^m$ is large (hot) the target distribution is flattened and the markov chain can explore the parameter space more freely, meaning they are less likely to get stuck in local modal points. After each time step, adjacent chains can swap Markov chain positions according to a probably given by the metropolis ratio:

$$\alpha_{m,m+1}(\theta_i^m, \theta_i^{m+1}) = \left(\frac{\pi(\theta_{i}^{m+1})}{\pi(\theta^{m}_i)}\right)^{1/T^m - 1/T^{m+1}}$$


This means that local modal points found by the explorative warmer chains, can be adjacently passed down to the colder chains which estimate the posterior distributions (in this algorithm the posterior distribution is estimated using samples from the coldest ($T_0 = 1$) chain only). Therefore, this algorithm allows for efficient mixing for disconnected and multimodal target distributions. 

The package also allows for two contemporary extenstions of standard Parallel Tempering markov chain Monte Carlo, an adaptive covariance matrix and adaptive temperature ladder. 


####i. Adaptive covariance matrix for each chain

The proposal distribution is given by 

$$
q(.|\theta_i^m) \sim \left\{\begin{array}{ll}
\mathcal{N}(\theta_i^m, \exp(\lambda_i)I_d), & i \leq i_{covar} ||  m>4 || p < 0.05  \\ 
\mathcal{N}(\theta_i^m, \exp(M_i)\Sigma_d),  & i > i_{covar} || p > 0.05    \\ 
\end{array} \right.
$$


* $\mathcal{N}(\theta_i^m, \lambda_tI_d)$ is a multivariate normal distibution with covariance matrix given by the identity ($I_d$) multiplied by an adaptive scalar value ($\lambda_i$) updated at each time step according to a stochastic approximation equation $\lambda_{i+1} = \lambda_i + \gamma_i(\alpha(\theta_i^m, \theta^*) - 0.234)$, $\gamma_i = (1+i)^{0.5}$)

* $\mathcal{N}(\theta_i^m,  M_i\Sigma_d)$ is a multivariate normal with the covariance matrix given by the the previous samples in the chain and the formula for $M_i$ is as given for $\lambda_i$ but with $\gamma_i = (1+i-i_{covar})^{0.5}$

* $i_{covar}$ is the number of runs which occur before the covavriance matrix starts estimating values. Having a large value here allows the markov chain to find a sensible place before the covariance matrix is estimated. 

For warmer chains ($m>4$) the adaptive covariance matrix is not used, as its use leads to very poor mixing. In the colder chains, after $i_{covar}$ steps, to ensure that the Markov chains converge to the target distribution, a sample from $\mathcal{N}(\theta_i^m,  M_i\Sigma_d)$ occurs 95% of the time, and a sample from $\mathcal{N}(\theta_i^m, \lambda_tI_d)$ occurs 5% of the time (i.e. $p \sim \mathcal{U}(0,1)$, as proposed in [Sherlock et al. 2010](https://projecteuclid.org/download/pdfview_1/euclid.ss/1290175840). 

The `adap_Cov` setting triggers the above proposal, if FALSE then only $\mathcal{N}(\theta_i^m, \lambda_t I_d)$ is used (default = TRUE). `adap_Covar_burn` is the value of $i_{covar}$ and `adap_Covar_freq` is the frequency they update (default = 1, every step).

####ii. Adaptive tempering ladder
The values of the temperature ladder ($T^m$) influence the rate at which adjacent markov chains are swapped. By allowing temperature values to change at each time step ($T_i^m$) according to the stochastic approximation as used for the proposal scaling paramters ($\lambda_i$ and $M_i$), it is possible force inter-chain swapping rates of 0.234 (as is optimal). The algorithm works by defining $T^m_0 := 10^{7m/(M-1)}$, $S^{m}_{0} := \log(T^{m+1}_{0} - T^{m}_{0})$
and updating by

$$S^{m}_{i+1} = S^m_i + \gamma_i(\alpha_{m, m+1}-0.234) $$
$$T_i^{m+1} = T_i^m + \exp(S_i^m) $$

The `adap_Temp` boolean variable in settings toggles with the temprature paraemters are on (default = TRUE), with the `adap_Temp_freq` variable stating how often the adaptive temperature ladder updates (default = 1, every step).


###Further reading

For a fuller understanding of Parallel Tempering MCMC algorithms I recommend:

* [Chapter 6 of Advance MCMC](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470669723) provides a good introduction to all population type chains, including PTMCMC. Also [Syed et al. 2019](https://arxiv.org/pdf/1905.02939.pdf)
* [Adaptive covariance PTMC](https://www.cs.ubc.ca/~nando/540b-2011/projects/8.pdf)
* Adaptive Temperature Ladder PTMC [Miasojedow et al. 2012](https://arxiv.org/pdf/1205.1076.pdf), [Vousden et al. 2015](https://doi.org/10.1093/MNRAS/STV2422)


## Implementation

See vingette.

## Performance

See ptmc_performance repository.

## License

See LICENCE file.

