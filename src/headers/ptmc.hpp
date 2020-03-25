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

namespace ptmc{
    
    template <typename D> // have no choice for parameter type
    struct PTMC
    {
        PTMC() {}
        
        VectorXd nonadaptiveScalar, adaptiveScalar;
        MatrixXd nonadaptiveCovarianceMat, adaptiveCovarianceMat;
        MatrixXd currentSample, currentSampleMean;
        MatrixXd posteriorOut;
        VectorXd currentLogPosterior, proposalSample;
        VectorXd temperatureLadder, temperatureLadderParameterised;
        
        int iterations, posteriorSamplesLength, thin, burninPosterior, burninAdaptiveCov, consoleUpdates, updatesAdaptiveCov, updatesAdaptiveTemp;
        int numberTempChains, numberFittedPar;
        int workingChainNumber, workingIteration;
        bool onDebug, onAdaptiveCov, onAdaptiveTemp;
        bool isSampleAccepted, isProposalAdaptive;

        VectorXd counterFuncEval, counterAccepted, counterPosterior ,counterAdaptive;
        VectorXd counterNonAdaptive, counterFuncEvalTemp, counterAcceptTemp;
        
        double proposedLogPosterior, alpha;
        D dataLogLikelihood;
        
        std::function<VectorXd()> samplePriorDistributions;
        std::function<double(VectorXd)> evaluateLogPrior;
        std::function<double(D,VectorXd)> evaluateLogLikelihood;

        double evalLogPosterior(VectorXd param)
        {
            double logPrior = this->evaluateLogPrior(param);
            if (isinf(logPrior))
                return log(0);
          
            double logLikelihood = this->evaluateLogLikelihood(this->dataLogLikelihood, param);
            return logPrior + logLikelihood;
        }
        
        void initialiseClass(D data, List settings)
        {
            this->dataLogLikelihood = data;
            this->numberTempChains = settings["numberTempChains"];
            this->numberFittedPar = settings["numberFittedPar"];
            this->iterations = settings["iterations"];
            this->thin = settings["thin"];
            this->burninPosterior = settings["burninPosterior"];
            this->burninAdaptiveCov = settings["burninAdaptiveCov"];
            this->consoleUpdates = settings["consoleUpdates"];
            this->onAdaptiveCov = settings["onAdaptiveCov"];
            this->onAdaptiveTemp = settings["onAdaptiveTemp"];
            this->updatesAdaptiveCov = settings["updatesAdaptiveCov"];
            this->updatesAdaptiveTemp = settings["updatesAdaptiveTemp"];
            this->onDebug = settings["onDebug"];
            
            this->counterFuncEval = VectorXd::Zero(this->numberTempChains);
            this->counterAccepted = VectorXd::Zero(this->numberTempChains);
            this->counterPosterior = VectorXd::Zero(this->numberTempChains);
            this->counterAdaptive = VectorXd::Zero(this->numberTempChains);
            this->counterNonAdaptive = VectorXd::Zero(this->numberTempChains);
            this->counterFuncEvalTemp = VectorXd::Zero(this->numberTempChains);
            this->counterAcceptTemp = VectorXd::Zero(this->numberTempChains);
            
            this->posteriorSamplesLength = (this->iterations-this->burninPosterior)/(this->thin);
            this->posteriorOut = MatrixXd::Zero(this->numberTempChains*this->posteriorSamplesLength,this->numberFittedPar+3);
            
            this->currentLogPosterior = VectorXd::Zero(this->numberTempChains);
            this->currentSampleMean = MatrixXd::Zero(this->numberTempChains,this->numberFittedPar);
            this->currentSample = MatrixXd::Zero(this->numberTempChains,this->numberFittedPar);
            
            this->proposalSample = VectorXd::Zero(this->numberFittedPar);
            this->nonadaptiveScalar = VectorXd::Zero(this->numberTempChains);
            this->adaptiveScalar = VectorXd::Zero(this->numberTempChains);
            
            this->nonadaptiveCovarianceMat = MatrixXd::Zero(this->numberFittedPar ,this->numberFittedPar );
            this->adaptiveCovarianceMat = MatrixXd::Zero(this->numberTempChains*this->numberFittedPar ,this->numberFittedPar );
            this->temperatureLadder =  VectorXd::Zero(this->numberTempChains);
            this->temperatureLadderParameterised =  VectorXd::Zero(this->numberTempChains-1);
            
            VectorXd initialSample;
            double initialLogLikelihood;
            
            for (int chainNum = 0; chainNum < this->numberTempChains; chainNum++){
                initialSample = this->samplePriorDistributions();
                initialLogLikelihood = this->evalLogPosterior(initialSample);
                while(isinf(initialLogLikelihood) || isnan(initialLogLikelihood)){
                    initialSample = this->samplePriorDistributions();
                    initialLogLikelihood = this->evalLogPosterior(initialSample);
                }
                
                this->currentSample.row(chainNum) = initialSample;
                this->currentSampleMean.row(chainNum) = initialSample;
                this->currentLogPosterior(chainNum) = initialLogLikelihood;
                
                this->nonadaptiveScalar(chainNum) = log(0.1*0.1/(double)this->numberFittedPar);
                this->adaptiveScalar(chainNum) = log(2.382*2.382/(double)this->numberFittedPar);
                
                temperatureLadder[chainNum] = pow(10, 7.0*(chainNum)/(numberTempChains-1.0));
                if (chainNum > 0)
                    temperatureLadderParameterised[chainNum-1] = log(temperatureLadder[chainNum]-temperatureLadder[chainNum-1]);
            }
            
            for(int parNum = 0; parNum < this->numberFittedPar ; parNum++){
                this->nonadaptiveCovarianceMat(parNum,parNum) = 1.0;
                for (int chainNum = 0; chainNum < this->numberTempChains; chainNum++){
                    this->adaptiveCovarianceMat(chainNum*this->numberFittedPar+parNum,parNum) = 1.0;
                }
            }
        }
        
        MatrixXd runPTMCC()
        {
            for (int i = 0; i < this->iterations; i++){
                this->workingIteration = i;
                updateAllChainsAndTemp();
            }
            return this->posteriorOut.block(0, 0, this->posteriorSamplesLength, this->numberFittedPar+3);
        }
        
        void updateAllChainsAndTemp()
        {
            for (int n = 0; n < numberTempChains; n++){
                this->workingChainNumber = n;
                getAcceptanceRate();
                updateSampleAndLogPosterior();
                updateOutputPosterior();
                updateProposal();
            }
            swapTemperatureChains();
            consoleUpdatefunction();
        }

        void getAcceptanceRate()
        {
            this->isSampleAccepted = false;
            selectProposalDist();
            this->proposedLogPosterior = this->evalLogPosterior(this->proposalSample);
            evaluateMetropolisRatio();
            this->counterFuncEval[this->workingChainNumber]++;
        }
        
        void selectProposalDist(){
            int numTempChainsNonAdaptive = 4;
            if (this->workingIteration < this->burninAdaptiveCov || uniformContinuousDist(0, 1) < 0.05 || !this->onAdaptiveCov || this->workingChainNumber > numTempChainsNonAdaptive){
                generateSampleFromNonAdaptiveProposalDist();
            }
            else{
                generateSampleFromAdaptiveProposalDist();
            }
        }
        
        void generateSampleFromNonAdaptiveProposalDist()
        {
            this->counterNonAdaptive[this->workingChainNumber]++; this->isProposalAdaptive = false;
            EigenMultivariateNormal normX_solver(this->currentSample.row(this->workingChainNumber), exp(this->nonadaptiveScalar[this->workingChainNumber])*this->nonadaptiveCovarianceMat, true, engine());
            this->proposalSample = normX_solver.samples(1);
        }
        
        void generateSampleFromAdaptiveProposalDist()
        {
            this->counterAdaptive[this->workingChainNumber]++; this->isProposalAdaptive = true;
            MatrixXd covarianceMatPerChain;
            covarianceMatPerChain = this->adaptiveCovarianceMat.block(this->workingChainNumber*this->numberFittedPar, 0, this->numberFittedPar, this->numberFittedPar);
            EigenMultivariateNormal normX_solver(this->currentSample.row(this->workingChainNumber), exp(this->adaptiveScalar[this->workingChainNumber])*covarianceMatPerChain, true, engine());
            this->proposalSample = normX_solver.samples(1);
        }
        
        void evaluateMetropolisRatio()
        {
            if(std::isnan(this->proposedLogPosterior - this->currentLogPosterior[this->workingChainNumber]))
                this->alpha = 0;
            else
                this->alpha = min(1.0, exp((this->proposedLogPosterior - this->currentLogPosterior[this->workingChainNumber])/this->temperatureLadder[this->workingChainNumber]));
        }
        
        void updateSampleAndLogPosterior()
        {
            if (uniformContinuousDist(0, 1) < this->alpha) {
                this->isSampleAccepted = true; this->counterAccepted[this->workingChainNumber]++;
                this->currentSample.row(this->workingChainNumber) = proposalSample;
                this->currentLogPosterior(this->workingChainNumber) = proposedLogPosterior;
            }
        }
        
        void updateOutputPosterior()
        {
            if ((this->workingIteration > (this->burninPosterior-1)) && (this->workingIteration%thin == 0) ) {
                int m = this->workingChainNumber;
                int l = this->posteriorSamplesLength;
                for (int p = 0; p < this->numberFittedPar; p++)
                    this->posteriorOut(m*l+this->counterPosterior[m], p) = this->currentSample(m,p);
                
                this->posteriorOut(m*l+this->counterPosterior[m], this->numberFittedPar) = this->currentLogPosterior(m);
                this->posteriorOut(m*l+this->counterPosterior[m], this->numberFittedPar+1) = this->temperatureLadder[m];
                this->posteriorOut(m*l+this->counterPosterior[m], this->numberFittedPar+2) = (double)this->counterAccepted[m]/(double)this->counterFuncEval[m];
                this->counterPosterior[m]++;
            }
        }
        
        void updateProposal()
        {
            int m = this->workingChainNumber;
            int P = this->numberFittedPar;

            if (this->isProposalAdaptive){
                this->adaptiveScalar[m] += pow(1+this->counterAdaptive[m],-0.5)*(this->alpha - 0.234);
                errorCheckNumberValid(this->adaptiveScalar[m]);
            }
            else{
                this->nonadaptiveScalar[m] += pow(1+this->counterNonAdaptive[m],-0.5)*(this->alpha - 0.234);
                errorCheckNumberValid(this->nonadaptiveScalar[m]);
            }
            
            if(((this->workingIteration)%(this->updatesAdaptiveCov) == 0) && (this->workingIteration > this->burninAdaptiveCov)){
                int iPosterior = (this->workingIteration-this->burninAdaptiveCov);
                double gainFactor = pow(1+iPosterior,-0.5);
                if (iPosterior == this->updatesAdaptiveCov){
                    this->currentSampleMean.row(m) = this->currentSample.row(m);
                    this->adaptiveScalar[m] = this->nonadaptiveScalar[m]*this->adaptiveScalar[m];
                }
                else{
                    this->currentSampleMean.row(m) = this->currentSampleMean.row(m) + gainFactor*(this->currentSample.row(m)-this->currentSampleMean.row(m));
                    this->adaptiveCovarianceMat.block(m*P,0,P,P) = this->adaptiveCovarianceMat.block(m*P,0,P,P) + gainFactor*((this->currentSample.row(m)-this->currentSampleMean.row(m)).transpose()*((this->currentSample.row(m))-(this->currentSampleMean.row(m)))) - gainFactor*this->adaptiveCovarianceMat.block(m*P,0,P,P);
                }
            }
        }
        
        void swapTemperatureChains()
        {
            if (this->workingIteration%updatesAdaptiveTemp == 0){
                for (int m = 0; m < this->numberTempChains; m++){
                    int p = uniformDiscreteDist(0, this->numberTempChains - 2);
                    this->counterFuncEvalTemp[p] ++;
                    int alphaTemp = evaluateMetropolisRatioTemp(p);
                    if (uniformContinuousDist(0, 1) < alphaTemp){
                        this->counterAcceptTemp[p]++;
                        swapSamplesandLogPosterior(p);
                    }
                    this->temperatureLadderParameterised[p] += pow((1+this->counterFuncEvalTemp[p]),(-0.5))*(alphaTemp - 0.234);
                    errorCheckNumberValid(this->temperatureLadderParameterised[m+1]);
                }
                if (this->onAdaptiveTemp){
                    for (int m = 0; m < this->numberTempChains-1; m++){
                        changeTemperature();
                        errorCheckTempNumberValid(this->temperatureLadder[m+1]);
                    }
                }
            }
        }
        
        void changeTemperature()
        {
            int m = this->workingChainNumber;
            this->temperatureLadder[m+1] = this->temperatureLadder[m] + exp(this->temperatureLadderParameterised[m]);
        }
        
        void errorCheckNumberValid(double value)
        {
            if (isinf(value)||isnan(value)){
                Rcout << "HERE IN THIS BIT" << endl;
                stop("Value: ", value);
            }
        }
        
        void errorCheckTempNumberValid(double temp)
        {
            if (temp < 1 ||isinf(temp)||isnan(temp)){
                Rcout << "HERE IN THIS BIT" << endl;
                stop("Value: ", temp);
            }
        }
        
        double evaluateMetropolisRatioTemp(int p)
        {
            double alphaTemp;
            if(std::isnan(this->currentLogPosterior[p+1] - this->currentLogPosterior[p]))
                alphaTemp = 0;
            else
                alphaTemp = min(1.0, exp((this->currentLogPosterior[p+1] - this->currentLogPosterior[p])*(1.0/this->temperatureLadder[p]-1.0/this->temperatureLadder[p+1])));
                        
            return alphaTemp;
        }
        
        void swapSamplesandLogPosterior(int p)
        {
            VectorXd swapSampleInterProp, swapSampleInterCurr;
            double swapLogPosteriorInterCurr, swapLogPosteriorInterProp;
            
            swapLogPosteriorInterCurr = this->currentLogPosterior(p); swapLogPosteriorInterProp = this->currentLogPosterior(p+1);
            this->currentLogPosterior(p) = swapLogPosteriorInterProp; this->currentLogPosterior(p+1) = swapLogPosteriorInterCurr;
            swapSampleInterCurr = this->currentSample.row(p); swapSampleInterProp = this->currentSample.row(p+1);
            this->currentSample.row(p) = swapSampleInterProp; this->currentSample.row(p+1) = swapSampleInterCurr;
        }
                
        double uniformContinuousDist(double minValue, double maxValue)
        {
            boost::random::uniform_real_distribution<> u(minValue,maxValue); return u(rng);
        }
        
        double uniformDiscreteDist(int minValue, int maxValue)
        {
            boost::random::uniform_int_distribution<> u(minValue, maxValue); return u(rng);
        }
            
        void consoleUpdatefunction()
        {
            int i = this->workingIteration;
            if(i%this->consoleUpdates == 0 && !this->onDebug)
                Rcout << "Running MCMC-PT iteration number: " << this->workingIteration << " of " <<  this->iterations << ". Current logpost: " << this->currentLogPosterior(0) << ". " << this->currentLogPosterior(1) << "           " << "\r";
        }
    };
};
// namespace ptmc
#endif
