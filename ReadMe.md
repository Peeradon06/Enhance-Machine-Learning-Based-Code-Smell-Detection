# Enhance Machine Learning-Based Code Smell Detection Through Hyper-Parameter Optimization

## About
This repository provides an implementation of hyper-parameter optimization for enhance machine learning-based code smell detection, as described in the paper:
**Enhance Machine Learning-Based Code Smell Detection Through Hyper-Parameter Optimization**

To preserve software quality and maintainability, machine learning-based code smell detection has been proposed and the results are promising. However, there are a lack of studies on applying hyper-parameter optimization (HPO) to improve the efficiency of code smell detection. Moreover, we found that there are a limited number of previous works indicating the effect of HPO to the code smell detection. 

This work proposed an enhanced version of machine learning-based code smell detection by applying hyper-parameter optimization techniques.

### Paper 
Enhance Machine Learning-Based Code Smell Detection Through Hyper-Parameter Optimization

### Machine Learning Algorithms 
* Decision tree (DT)
* Random forest (RF)

### Hyper-parameter Optimization Techniques 
* Particle Swarm Optimization (PSO)
* Bayesian Optimization with Tree-structure Parzen Estimator (BO-TPE)
* Bayesiam Optimization with Random forest (BO-RF/SMAC)

### Requirements
* Python 3.5+
* [scikit-learn](https://scikit-learn.org/stable/)
* [hyperopt](https://github.com/hyperopt/hyperopt)
* [optunity](https://github.com/claesenm/optunity)
* [SMAC3](https://github.com/automl/SMAC3)
* Noted: the SMAC3 library currently support only linux/unix operating system.

## Citation
If you find this repository useful or being used within your research, please cite this paper as:

P. Sukkasem, C. Soomlek, "Enhance Machine Learning-Based Code Smell Detection Through Hyper-Parameter Optimization," 2023 2oth International Joint Conference on Computer Science and Software Engineering (JCSSE), Phitsanulok, Thailand, 2023.
