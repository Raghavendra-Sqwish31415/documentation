# Multi-ArmedBandit Algorithms

- [Python Library Contextualbandits](#Python-Library-Contextualbandits)
- [Contextual Multi-Armed Bandits (CMAB)](#Contextual-Multi-Armed-Bandits-CMAB) - Speedup for UCB
- [Decision Language Model](#Decision-Language-Model) - LLM Generates the Reward Function
- [LinUCB](#LinUCB)
- [Adaptive Question Answering (Using LinUCB)](#Adaptive-Question-Answering-Using-LinUCB)
- [Thompson Sampling](#Thompson-Sampling)
- [NeuralUCB](#NeuralUCB)
- [Online Personalizing](#Online-Personalizing)
- [Synthetic Dataset to Jumpstart Your Bandits](#Synthetic-Dataset-to-Jumpstart-Your-Bandits)
- [Personalised Prompt Using DPO](#Personalised-Prompt-Using-DPO)
- [Neural Contextual Bandits](#Neural-Contextual-Bandits)
- [ByteDance Online Recommendation System](#ByteDance-Online-Recommendation-System)
- [Optimizely](#Optimizely)

## Python Library [contextualbandits](https://contextual-bandits.readthedocs.io/en/latest/)
This is a Python library providing implementations of a broad range of contextual-bandit algorithms. The first from each type is recommended. 
- **Author**: David Cortes  [paper](https://arxiv.org/pdf/1811.04383)
- **Publication Date**: 23 Nov 2019

### Summary of paper based on Repo
The key contribution is using a bootstrapping method to approximate UCB instead of hardcoding the variance term. This is done by just obtaining the reward with multiple samples. 

1. Each of the $K$ arms is actually $m$ different reward models.
2. when a context arrives, for each arm, $m$ different rewards are computed (from each model) and the 80 percentile (just for example) reward is used. 
3. The action that gives the best UCB is chosen and new rewards are observed.
4. This and the context are used to update all $m$ reward models.   

### Pseudocode

$$
\begin{align}
&\textbf{Inputs:} \quad m \text{ (number of resamples)}, \quad p \text{ (percentile)}, \quad \hat{f}_{1:k, 1:m} \text{ (online oracles)} \\
&\textbf{For\ each\ round} \quad t = 1, 2, \dots \quad \textbf{do:} \nonumber \\
&\quad \text{Receive context } \quad x_t \in \mathbb{R}^d \nonumber \\
&\quad \textbf{For\ each\ arm} \quad q = 1, \dots, k: \nonumber \\
&\quad\quad \hat{r}_{\text{ucb}, q} = \text{Percentile}_p \left\{ \hat{f}_{q,1}(x_t), \dots, \hat{f}_{q,m}(x_t) \right\} \\
&\quad \text{Choose action } \quad a = \arg\max_q \hat{r}_{\text{ucb}, q} \\
&\quad \text{Play arm } \quad a \quad \text{ and observe reward } \quad r_t \in \{0, 1\} \nonumber \\
&\quad \textbf{For\ each\ resample} \quad s = 1, \dots, m: \nonumber \\
&\quad\quad w \sim \text{Gamma}(1, 1) \\
&\quad\quad \text{Update } \quad \hat{f}_{a, s} \quad \text{ with example } \quad (x_t, r_t) \quad \text{ and weight } \quad w \nonumber
\end{align}
$$

### Cold-start problem
This is addressed by a mixture of these 2 methods:
1. initially sampling rewards before using the contexts to carry out actions and obtaining proper rewards.
2. Return a smoothed reward with magic smoothing factors suggested in the paper.

$$
\begin{align}
&\textbf{Inputs:} \quad a, b, m \quad (\text{hyperparameters}), \quad \text{mode} \in \{\text{mab}, \text{smooth}\} \\
&\textbf{Given:} \quad R_k = \{(x_i, r_i)\} \text{ for arm } k, \quad \pi_k(x) \text{ (contextual policy)} \\
&n = |R_k|, \quad n_1 = |\{r_i = 1\}|, \quad n_0 = |\{r_i = 0\}| \\
\\
&\textbf{If\ mode} = \text{``mab''} \textbf{:} \nonumber \\
&\quad \textbf{If} \quad n_0 < m \quad \textbf{or} \quad n_1 < m: \nonumber \\
&\qquad \hat{r}_k \sim \text{Beta}(a + n_1, \quad b + n_0) \\
&\quad \textbf{Else:} \quad \hat{r}_k = \pi_k(x) \\
\\
&\textbf{If\ mode} = \text{``smooth''} \textbf{:} \nonumber \\
&\quad \hat{r}_k = \pi_k(x) \\
&\quad \hat{r}_{\text{smooth}} = \frac{n \cdot \hat{r}_k + a}{n + b} \\
&\quad \hat{r}_k \leftarrow \hat{r}_{\text{smooth}} \\
\\
&\textbf{Return:} \quad \hat{r}_k \nonumber
\end{align}
$$

### Online Algorithms

#### Upper Confidence Bound Variants
- **BootstrappedUCB** (Robust)
- **PartitionedUCB**
- **LogisticUCB**
- **LinUCB**

#### Thompson Sampling Variants
- **BootstrappedTS**
- **PartitionedTS**
- **ParametricTS**
- **LogisticTS**
- **LinTS**

#### Randomized
- **AdaptiveGreedy** (Best performing)
- **SoftmaxExplorer**
- **EpsilonGreedy**
- **ExploreFirst**

#### Active Learning
- **ActiveExplorer** 
- **AdaptiveGreedy**

#### Naive
- **SeparateClassifiers**

## Contextual Multi-Armed Bandits (CMAB)
### Author
- **Organisations**: University of Toronto, University of Alberta, Google, Inc.  [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37042.pdf)
- **Publication Date**: 2010

### Goal
Select the most relevant ad for a user's query to maximize clicks (CTR).

### CMAB setup

- contexts: user queries
- actions: ads
- reward/action value function: click-though-rate (CTR) 

CTR satisfies a Lipschitz condition. This means that for queries that are similar and ads that are similar, the click-though-rates are also similar.

**Note**: This algorithm is not optimised for changing user preferences as it assumes stationary value function (user prefernece/click-though-rates/payoffs). It is a good starting point and maybe even a **good enough approximate**. 

### Mathematical formulation

- Let $X$ be a metric space of queries (contexts).
- Let $Y$ be a metric space of ads (actions).
- The expected payoff function $\mu: X \times Y \rightarrow [0, 1]$ is **Lipschitz**:

$$
|\mu(x, y) - \mu(x', y')| \leq L_X(x, x') + L_Y(y, y')
$$

- At each round $t$, a query $x_t \in X$ is observed.
- The algorithm selects an ad $y_t \in Y$, receives a payoff $\hat{\mu}_t \in [0, 1]$ such that:

$$
\mathbb{E}[\hat{\mu}_t] = \mu(x_t, y_t)
$$


### Regret

The regret measures the performance gap between the algorithm and the Bayes optimal strategy (which always picks the best ad for each query):

$$
R_A(T) = \sum_{t=1}^{T} \left[ \sup_{y \in Y} \mu(x_t, y) - \mathbb{E}[\mu(x_t, y_t)] \right]
$$

The goal is to design an algorithm with sublinear regret $R_A(T) = o(T)$.

### Query-Ad Clustering Algorithm

The paper introduces a **query-ad-clustering algorithm**, which:

- Clusters the query space $X$ into balls of decreasing radius over phases.
- Gets a new query and classifies it into a specific cluster.
- Uses Upper Confidence Bound (UCB) to balance exploration vs. exploitation.

Here is pseudocode to implement the algorithm practically:

### Pseudocode: Continuous/Real Time learning
$$
\begin{aligned}
&\textbf{INPUT:} \quad I, \quad a, \quad b\\
&\textbf{for} \quad \mathrm{phase\_index}=1 \quad \text{ to } \quad I \quad \textbf{do}\\
&\quad R \gets 2^{\,\mathrm{phase\_index}(-a - b - 4)}\\
&\quad\text{cluster queries into balls of radius } \quad R\\
&\quad Y_0 \gets\{\,\text{ads all pairwise within } \quad R\}\\
&\quad\textbf{classify\ new\ query} \quad x_t \quad \text{ into cluster } \quad C_j\\
&\quad n[y]\gets 0 \quad (\forall y\in Y_0)\\
&\quad\textbf{for} \quad t=1 \quad \text{ to } \quad 2^{\mathrm{phase\_index}} \quad \textbf{do}\\
&\quad\quad\displaystyle \mathrm{score}[y]\gets \mu^{C_j}(y)+\sqrt{\frac{4\,\mathrm{phase\_index}}{1+n[y]}} \quad (\forall y)\\
&\quad\quad y_t\gets\arg\max_y\mathrm{score}[y]\\
&\quad\quad\text{display } \quad y_t\\
&\quad\quad\textbf{if} \quad \text{add clicked} \quad \textbf{then} \\
&\quad\quad\quad n[y_t] \leftarrow n[y_t] + 1 \\
&\quad\quad\textbf{end\ if}\\
&\quad\quad \mu^{C_j}(y) \leftarrow \frac{n[y]}{t} \quad (\forall y)\\
&\quad\textbf{end\ for}\\
&\textbf{end\ for}\\
\end{aligned}
$$

```pseudo
INPUT:
    I      # total number of phases
    a      # dimensionality of query embeddings
    b      # dimensionality of ad embeddings

FOR phase_index FROM 1 TO I DO

    # --- Step 1: Determine clustering radius for this phase
    COMPUTE R AS 2^(phase_index × (–a – b – 4))

    # --- Step 2: Group similar queries together
    SPLIT the entire query embedding space INTO clusters 
        SUCH THAT each cluster has radius at most R
    LABEL these clusters as {Cluster_1, Cluster_2, …}

    # --- Step 3: Choose a compact set of ads
    SELECT a subset Y0 of ads 
        SO THAT every pair of ads in Y0 is within distance R
    # These are the only ads we will consider displaying this phase

    # --- Step 4: Determine the cluster that a new query belongs to. 
    # --- Run a UCB-based strategy for that cluster
    CLASSIFY query x_t into Cluster_j

    # Initialize the number of times each ad has been shown
    FOR each ad y IN Y0 DO
        SET n[y] ← 0
    END FOR

    # Repeat for the total number of time steps in this phase
    FOR t FROM 1 TO 2^phase_index DO

        # Compute the UCB score for each ad:
        #   base_estimate = u^{Cluster_j}(y)
        #   bonus = sqrt( (4 × phase_index) / (1 + n[y]) )
        FOR each ad y IN Y0 DO
            CALCULATE score[y] = u^{Cluster_j}(y) 
                               + sqrt((4 * phase_index) / (1 + n[y]))
        END FOR

        # Pick the ad with the highest UCB score
        CHOOSE y_t = argmax_y score[y]

        # Display the chosen ad to users whose queries fall in Cluster_j
        DISPLAY y_t TO users_in Cluster_j

        # Observe whether the ad was clicked (1) or not (0)
        OBSERVE click_signal ∈ {0, 1}

        # Update the count of how many times y_t has been shown
        IF ad clicked THEN
            n[y_t] ← n[y_t] + 1
        END IF

        # Update the value/pay-off/reward function (click though rate)
        FOR each ad y IN Y0 DO
            UPDATE u^{Cluster_j}(y) = n[y]/t
        END FOR

    END FOR  # end time steps

END FOR  # end phase loop
```

#### Regret Bound

Let:
- $a, b$: covering dimensions of $X$ and $Y$
- $\tilde{a}, \tilde{b}$: packing dimensions of $X$ and $Y$

Then for any $\varepsilon > 0$, the regret is:

- **Upper bound**:

$$
R_A(T) = O\left(T^{\frac{a+b+1}{a+b+2} + \varepsilon}\right)
$$

- **Lower bound** (for any algorithm):

$$
R_A(T) = \Omega\left(T^{\frac{\tilde{a} + \tilde{b} + 1}{\tilde{a} + \tilde{b} + 2} - \varepsilon}\right)
$$

For **finite or bounded Euclidean spaces**, covering and packing dimensions are equal, yielding **tight bounds**.

---

## Decision Language Model
### Author
- **Organisations**: Deepmind, MIT, Harvard  [paper](https://arxiv.org/html/2409.13447v2)
- **Publication Date**: 23 September 2024

### Summary
![Screenshot 2025-07-04 at 12.14.52](https://hackmd.io/_uploads/SkphwNrSee.png)
1. **Context Preparation**  
   Provide the LLM with three context descriptions:
   - A language command
   - A list of per-arm context/features that can be used for reward functions
   - Relevant codebase

2. **Reward Proposal**  
   The LLM proposes candidate reward functions based on the provided context.

3. **Policy Training**  
   Use the candidate reward functions to train multiple multi-armed bandit (one for each proposed reward).

4. **Simulation and Outcome Comparison**  
   Simulate the bandits to generate outcome comparisons.

5. **LLM Self-Reflection**  
   Query the LLM to perform self-reflection by selecting the best candidate reward that aligns with the original intent.

## LinUCB
### Author
- **Organisations**: Yahoo  [paper](https://arxiv.org/pdf/1003.0146)
- **Publication Date**: 1 March 2012
### Key Idea
The main contribution of this paper is an alternative way of calculating the  Upper Confidence Bound algorithm used by Google CMAB. It assumes that the reward is a linear map from the context:

$$
\mathbb{E}\bigl[r_{a}\mid x\bigr] = x^\top \theta_a.
$$

Practically, we will also need to train (continuously update from reward signals) a covariance matrix $A_a$ along with the linear parameter $\theta_a$.

Let a be the action taken. (In google CMAB, y referred to action). The linear map from the context to the UCB score contains a confidence interval term. The best action is chosen as follows:

$$
\begin{align}
\text{score}(a) &= \theta_a^T x + \sqrt{x^T A_a^{-1} x^T} \\
a_\text{best} &= \arg\max_a \quad \text{score}(a)
\end{align}
$$

Once the best action is taken, the reward signal updates the parameters in the following way:

$$
\begin{align}
\theta_{a_\text{best}} &= \theta_a^T x + x^T A_a^{-1} x^T \\
a_\text{best} &= \arg\max_a \quad \text{score}(a) \quad \rightarrow \quad \text{reward} = r \\
b_a &\leftarrow b_a + r x\\
A_a &\leftarrow A_a + xx^T \\
\theta_{a_\text{best}} &\leftarrow A_a^{-1} r
\end{align}
$$

### Pseudocode

```pseudo
# --- LinUCB Contextual Bandit

# 1. Initialise the covariance matrix and the signal response vector
FOR each arm a in 1,...,K DO
    A_a ← I_d         # dxd identity matrix
    b_a ← 0_d         # zero vector
END FOR

FOR steps i = 1 TO T DO
    Observe context vectors x_i: a = 1,...,K

    FOR each arm a = 1,...,K DO
        # Compute linear parameter
        theta_a ← inv(A_a) @ b_a 
        
        # Compute UCB (value or reward)
        UCB_a,i ← (theta_a).T @ x_i + alpha * mahanobolis norm(x_i) with cov=A_a
    END FOR

    a_i ← ARGMAX_a UCB_a,i    # pick arm with highest UCB
    Play arm a_i and observe reward r_i

    # Update statistics for chosen arm
    A_(a_i) ← A_(a_i) + outer_product(x_i, x_i)
    b_(a_i) ← b_(a_i) + r_i * x_i
END FOR
```

## Adaptive Question Answering (using LinUCB)
### Author
- **Organisations**: Radboud University, Nijmegen, The Netherlands; University of Amsterdam, Amsterdam, The Netherlands  [paper](https://arxiv.org/html/2409.13447v2)
- **Publication Date**: 23 September 2024
### Summary
Implements LinUCB but penalises reward for computation cost. Does classification. 

## Thompson Sampling
- **Organisations**: [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/paper-28.pdf)
- **Publication Date**: 2016

## NeuralUCB
- **Organisations**: Google [paper](https://arxiv.org/pdf/1911.04462)
- **Publication Date**: 2020

Basically, a reward model is trained from the physical rewards (online learning). 

### Pseudocode
Algorithm (NeuralUCB, K arms)

Input: $T, \quad \lambda, \quad \nu, \quad \delta, \quad S, \quad \eta, \quad J, \quad m, \quad L$.

Initialize: $\theta_0\sim\text{Gaussian}, \quad Z_0=\lambda I.$

$$
\begin{align}
&\textbf{For} \quad t=1,\dots,T \quad \textbf{do:} \\
&\quad \text{Observe contexts } \quad \{x_{t,a}\}_{a=1}^K.\\
&\quad \textbf{For} \quad a=1,\dots,K \quad \textbf{do:}\\
&\quad \quad U_{t,a} = f(x_{t,a};\theta_{t-1}) + \gamma_{t-1}\sqrt{\frac{g(x_{t,a};\theta_{t-1})^\top Z_{t-1}^{-1}\,g(x_{t,a};\theta_{t-1})}{m}}. \quad (g \text{ is the gradient of } f )\\
&\quad \textbf{end\ for}\\
&\quad a_t=\arg\max_{a}U_{t,a}, \quad \text{play } \quad a_t, \quad \text{observe } \quad r_{t,a_t}.\\
&\quad Z_t = Z_{t-1} + \frac{1}{m}\,g(x_{t,a_t};\theta_{t-1})\,g(x_{t,a_t};\theta_{t-1})^\top,\\
&\quad \theta_t = \mathrm{TrainNN}\bigl(\lambda, \quad \eta, \quad J, \quad m, \quad \{(x_i,a_i)\}_{i=1}^t, \quad \{r_{i,a_i}\}_{i=1}^t, \quad \theta_0\bigr),\\
&\quad \gamma_t = \text{(update as per line 13 of Alg. 1 of paper)}.
\end{align}
$$


## Online Personalizing
- **Organisations**: J.P. Morgan [paper](https://arxiv.org/abs/2404.16115)
- **Publication Date**: 2020

This paper optimizes a delta to the input prompt embedding vector to improve the prompt. The NeuralBandit is either NeuralUCB or NeuralTS.
![Screenshot 2025-07-07 at 12.41.47](https://hackmd.io/_uploads/BkukXVYrex.png)

## Synthetic Dataset to Jumpstart Your Bandits
The goal is to solve the coldstart problem where at the start, the bandit has very few rewards to learn from. LLMs generate a small synthetic dataset to pretrain the bandit before starting online learning.

**Note**: The bandit uses LinUCB for training the bandit. 
![Screenshot 2025-07-07 at 14.50.48](https://hackmd.io/_uploads/rJG6MUYSex.png)
- **Organisations**: Borealis AI [paper](https://arxiv.org/pdf/2406.19317v2) [code](https://github.com/BorealisAI/jump-starting-bandits/blob/main/pretrained-bandit.py)
- **Publication Date**: 29 Oct 2024


## Personalised Prompt using DPO
This bandit is a transformer that outputs 2 responses, which are ranked by users and then trained on that. 

Can this be done in our case? 
1. The transformer suggests 2 prompts

2. Both prompts are fed into an external LLM

3. LLM produces 2 responses --> 2 rewards

4. Higher reward --> Positive prompt ||| Lower reward --> Negative prompt

5. Use DPO, train transformer

- **Organisations**: Stanford and University of Toronto [paper](https://arxiv.org/pdf/2410.14001)
- **Publication Date**: 5 May 2023

=================================================================
HAVE NOT READ YET


## Neural Contextual Bandits
- **Organisations**: University of Illanois [paper](https://arxiv.org/pdf/2305.03784)
- **Publication Date**: 5 May 2023

## ByteDance online recommendation System
- **Organisations**: ByteDance [paper](https://www.cs.princeton.edu/courses/archive/spring21/cos598D/icde_2021_camera_ready.pdf)
- **Publication Date**: 2021

## [Optimizely](https://www.optimizely.com/insights/blog/contextual-bandits-in-personalization/)

Uses decision trees to maximise business KPI