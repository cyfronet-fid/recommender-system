# Introduction
We are using AI-based algorithms to serve relevant recommendations. This file contains brief descriptions of the 2 algorithms used.

## NCF (Neural Collaborative Filtering)
This algorithm is a modern take on the classic Collaborative Filtering recommendation algorithm. It's easy to understand and implement. 

It uses the dataset of user-service pairs, which represent either positive (user liked the service) or negative correlation (user didn't like the service, or ignored it). This is later fed to the single MLP network with two inputs, one for the user and one for service, an arbitrary amount of hidden layers and one output, which predicts whether the user will like or dislike a given service. The network architecture can be seen in the following figure:
<p align="center">
    <img src="https://user-images.githubusercontent.com/30239467/142465251-96df19e0-7ed2-4f9e-921a-dc35b98f1fa7.png" alt="drawing" width="500"/>
</p>

Further reading:
- [The original NCF paper](https://arxiv.org/abs/1708.05031)
- [Great, step-by-step explanation](https://towardsdatascience.com/neural-collaborative-filtering-96cef1009401)


## TD3 (Twin Delayed Deep Deterministic Policy Gradient)
This is a state-of-the-art Reinforcement Learning algorithm. It is based on a widely used DDPG algorithm, with a small number of enhancements which drastically improve performance. The DDPG itself is based on the Actor-Critic RL architecture, one of the classic RL algorithms.
Our usage of the TD3 in the context of recommendation system is a fairly novel idea, only briefly mentioned in recent papers.

Every RL algorithm operates in some environment. In the context of recommendations, the environment represents the user's behaviour. To summarize, there are: 
- State (S) - some vector representation of the user, and his past interactions with the system
- Action (A) - recommendation served by the algorithm in the given state S, usually some set of services, represented in some vector space
- Reward (R) - a scalar (usually between 0 and 1), which is a numeric representation of whether the recommendation (A) is up to the user's liking

Now, the 2 major parts of the TD3 and DDPG are:
- Actor - which produces ("acts") recommendations A, in the given state S
- Critic - which judges ("criticizes") an action A, chosen by the actor in the state S

Both Actor and Critic are regular MLP networks and are trained according to the DDPG algorithm:
- The Actor's loss function tries to maximize the Critic's judgements
- The Critic's loss function tries to guide the Critic to more accurately judge actions, produced by the Actor.


This is all, of course, a general overview of the algorithm, meant to give you the big picture of how it's working. For further reading please refer to:
- [The original DDPG paper](https://arxiv.org/abs/1509.02971)
- [The original TD3 paper](https://arxiv.org/abs/1802.09477)
- [The RL "bible", Actor-Critic explained in the section 13.5](http://incompleteideas.net/book/the-book.html)
- [One of the best webpages regarding state-of-the-art RL, including TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)
