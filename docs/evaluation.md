# Introduction
This document describes the evaluation of our recommendation algorithms. Firstly on the neural network (low) level and then on the higher level.

## NCF
The evaluation of the network is done using well-known and widely used metrics:
- accuracy - specifies the number of correctly labeled examples, that is the examples that the algorithm correctly predicted to be of user's interest
- [f1 score](https://en.wikipedia.org/wiki/F-score)

## TD3
For now, the RL evaluation is based on the environment simulation, since at train-time we are not able to gather data from the real environment (it takes too long and would require deploying the experimental version to the production). 

The simulation is carried through as follows:
1. Each user's behaviour is approximated for as many steps as specified:
   1. Given the recommendation served by the evaluated algorithm, based on the user's historic data and personal information we were able to approximate the potential scale of interest in each of the served services 
   2. Based on the approximated interest, the rewards are drawn from the binomial distribution. The higher the interest the more likely it is to order the service and vice-versa. 
   3. Services that have scored a positive reward are added to the simulated user's history
2. The final rewards obtained by each user are averaged and used as a direct metric for evaluating the TD3 system.

## Higher level metrics
For the time being, we've concentrated on lower-level metrics, but additional high-level metrics are on the way. Furthermore, we use the modified hitrate metric, which is based on the user's clicks on the page where the recommendations are shown.

### Modified hitrate
Currently, the recommendations are served in the context of the search view in the [EOSC Marketplace portal](https://marketplace.eosc-portal.eu/services?object_id=&type=&anchor=&sort=_score&q=)

So, our modified hitrate is very simple and intuitive, it is the ratio of clicks on the recommendation panel to clicks on the list of the found services.