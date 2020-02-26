# Parallel Tempering Markov Chain Monte Carlo
##Introduction

This repository contains source files to easily run a Metropolis Hastings markov chain Monte Carlo with Parallel Tempering algorithm as outlined by Miasojedow et al. 2012 (https://arxiv.org/pdf/1205.1076.pdf). 

####Very brief overview of Parallel Tempering MCMC
Parallel Tempering runs *m* chains simulationeously with a likelihood function given by 

$$ $$

where T_0 = 1, and is a monotone increasing sequence in m. When T_m is small (i.e. cold) the chains run similar to a normal Metropolis Hasting algorihtm, however then T_m is large (hot) the target distribution is flattened and the markov chain can mix more freely. After each time step, the chains can swap parameter vectors through the metropolis ration value. This allows for efficient mixing for target distributions which are multimodal and discounnected. 

The code also allows for two contemporary extenstions of standard Parallel Tempering markov chain Monte Carlo namely

* i) Adaptive covariance matrix for each chain
* ii) Adaptive tempering ladder

####Further reading

For a fuller understanding of Parallel Tempering MCMC algorithms I reccommend

* PTMCMC Chapter 6 of Advance MCMC provides a good introduction to all population type chains, incluyding PTMCMC. Also https://arxiv.org/pdf/1905.02939.pdf
* Adaptive covariance PTMCMC: https://www.cs.ubc.ca/~nando/540b-2011/projects/8.pdf
* Adaptive covariance and temperature ladder PTMCMC: https://arxiv.org/pdf/1205.1076.pdf, https://doi.org/10.1093/MNRAS/STV2422


## Implementation

See vingette.

#### Settings
There are various settings which can be chosen, including:

* *run_l*: number of steps in markov chain (default = 100,000)


## License

See LICENCE.

