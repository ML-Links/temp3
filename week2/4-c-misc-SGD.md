# Gradient Descent → Stochastic Gradient Descent (SGD)

## 1. The Problem: Gradient Descent with Large Datasets

**Gradient Descent (GD)** is an optimization algorithm used to minimize a loss function by adjusting model parameters (weights and biases).

The objective is to find parameter values that reduce the difference between the model's predictions and the actual outcomes.

In standard **Batch Gradient Descent**, the gradient is calculated using the entire training dataset before updating the parameters.

For a dataset containing $n$ observations, the average loss gradient is:

$$
\nabla L(w)=\frac{1}{n}\sum_{i=1}^{n}\nabla L_i(w)
$$

Where:

- $w$: Model parameters (weights and biases).
- $n$: Number of training observations.
- $L_i(w)$: Loss associated with observation $i$.
- $\nabla L(w)$: Average gradient calculated using all training observations.

### Example: A Dataset with One Million Observations

Suppose we are training a neural network using **1,000,000 observations**.

With Batch Gradient Descent:

1. The model processes all 1,000,000 observations.
2. The gradients are calculated and averaged across all observations.
3. The model parameters are updated once.
4. The process repeats using the updated parameters.

This means that **one parameter update requires processing the entire dataset**.

### Why is this a problem?

For large datasets, calculating a full gradient can be computationally expensive.

- Each parameter update requires processing many observations.
- The time required for each update increases with dataset size.
- Training can become slow because parameter updates occur only after the entire dataset has been processed.

**Main limitation:** Batch Gradient Descent becomes expensive per update when datasets are very large.

*Note: The entire dataset does not necessarily need to be stored in memory at once. The main issue is the computational work required for each full-gradient update.*


## 2. The Solution: Stochastic and Mini-Batch Gradient Descent

To reduce the computational work required for each update, we can estimate the gradient using only a subset of the training observations.

This is the basic idea behind **Stochastic Gradient Descent (SGD)** and **Mini-Batch Gradient Descent**.

Instead of processing all $n$ observations before updating the weights, the algorithm uses a smaller group of observations called a *mini-batch*.

### Example: Using Mini-Batches

Suppose the dataset contains **1,000,000 observations**.

Instead of using all observations for one update, we select a mini-batch of 100 observations.

The training procedure becomes:

1. Select 100 observations.
2. Calculate the model's predictions and loss for those observations.
3. Calculate the average gradient for the mini-batch.
4. Update the model parameters.
5. Continue with another mini-batch.

The model can now update its parameters after processing only **100 observations instead of 1,000,000**.

This reduces the computational cost of an individual update and allows the model to update its parameters more frequently.

### Mathematical Representation

For a mini-batch $\mathcal{B}$ containing $m$ observations, define the average mini-batch loss:

$$
g_{\mathcal{B}}(w)=\frac{1}{m}\sum_{i\in\mathcal{B}}L_i(w)
$$

Its gradient is:

$$
\nabla g_{\mathcal{B}}(w)=\frac{1}{m}\sum_{i\in\mathcal{B}}\nabla L_i(w)
$$

The parameter update is:

$$
\boxed{w \leftarrow w-\alpha\nabla g_{\mathcal{B}}(w)}
$$

Where:

- $w$: Model parameters.
- $\alpha$: Learning rate, controlling the size of the update.
- $m$: Number of observations in the mini-batch.
- $\mathcal{B}$: Selected mini-batch.
- $\nabla g_{\mathcal{B}}(w)$: Gradient calculated using that mini-batch.

**Important:** The update rule is essentially the same as in Batch Gradient Descent. What changes is how the gradient is calculated.

In Batch Gradient Descent, the gradient uses all training observations. In mini-batch SGD, it uses only the selected mini-batch.

## 3. Batch Gradient Descent vs. Mini-Batch SGD vs. Pure SGD

There are three common approaches, distinguished by the number of observations used for each parameter update.

### A. Batch Gradient Descent

Uses the **entire training dataset** to calculate each gradient.

If the dataset contains 1,000 observations, the algorithm processes all 1,000 observations before updating the weights.

### B. Mini-Batch Gradient Descent

Uses a **small subset of observations** for each update.

For example, if a dataset contains 1,000 observations and the mini-batch size is 100, the model updates its weights after processing each group of 100 observations.

Common mini-batch sizes include 32, 64, 128, or 256, although other sizes may be appropriate.

### C. Pure Stochastic Gradient Descent

Uses **only one randomly selected observation** for each parameter update.

If the dataset contains 1,000 observations, the model can perform 1,000 updates while processing each observation once.

### Important Terminology

Strictly speaking:

- **Batch GD:** Uses all observations per update.
- **Mini-Batch GD:** Uses a subset of observations per update.
- **Pure SGD:** Uses one observation per update.

However, in modern deep learning, **SGD is commonly used to refer to mini-batch SGD**.

This is also the convention used in the lecture slides.


## 4. Understanding Iterations, Mini-Batches, and Epochs

These terms are important for understanding how SGD works.

- **Mini-batch:** A group of training observations used to calculate one parameter update.
- **Iteration (or step):** One parameter update.
- **Epoch:** One complete pass through the training dataset.

### Example: 1,000 Training Observations

Assume we shuffle the dataset and process each observation once per epoch, without replacement. Sampling independently with replacement is another SGD convention; a fixed number of draws need not visit every observation.

| Method | Observations per update | Updates per epoch |
|---|---:|---:|
| Batch GD | 1,000 | 1 |
| Mini-Batch SGD | 100 | 10 |
| Mini-Batch SGD | 50 | 20 |
| Pure SGD | 1 | 1,000 |

If the mini-batch size is 100:

$$
\text{Updates per epoch}=\frac{1000}{100}=10
$$

For a dataset containing 1,000,000 observations and a mini-batch size of 100:

$$
\text{Updates per epoch}=\frac{1,000,000}{100}=10,000
$$

Therefore, instead of performing one parameter update after processing the entire dataset, the model performs **10,000 parameter updates during each epoch, each based on 100 observations**.

More updates do not automatically mean faster overall convergence. The total training time also depends on the quality of the gradients, learning rate, and computational efficiency.


## 5. Numerical Example: How the Gradient Estimate Changes

Consider a simplified model with one parameter:

$$
w=10
$$

Assume the learning rate is:

$$
\alpha=0.1
$$

The dataset contains four observations. At the current parameter value, suppose their individual gradients are:

| Observation | Individual gradient |
|---|---:|
| 1 | 4 |
| 2 | 2 |
| 3 | −2 |
| 4 | 0 |

### A. Batch Gradient Descent

The gradient is the average of all four gradients:

$$
\nabla L(w)=\frac{4+2-2+0}{4}=1
$$

The parameter update is:

$$
w_{\text{new}}=10-0.1(1)=9.9
$$

### B. Mini-Batch SGD

Suppose the selected mini-batch contains observations 1 and 2.

The estimated gradient is:

$$
\nabla g_{\mathcal B}(w)=\frac{4+2}{2}=3
$$

The parameter update becomes:

$$
w_{\text{new}}=10-0.1(3)=9.7
$$

### C. Pure SGD

Suppose the randomly selected observation is observation 3.

Its gradient is:

$$
\nabla L_3(w)=-2
$$

The parameter update becomes:

$$
w_{\text{new}}=10-0.1(-2)=10.2
$$

### What Does This Example Show?

All three methods use the same basic update rule.

However, they produce different updates because they use different observations to estimate the gradient.

- Batch GD uses the average gradient across the entire dataset.
- Mini-Batch SGD uses an approximate gradient from a subset.
- Pure SGD uses the gradient of a single observation.

Notice that the pure SGD update moves in the opposite direction from the full-batch update in this example. This illustrates how individual gradients may differ from the average gradient.

**Random sampling or shuffling makes the batch selection, and therefore the gradient estimate, vary between updates. An individual estimate can differ from the full-data gradient.**

*These values are illustrative. In actual training, gradients are recalculated after each parameter update.*



## 6. How Mini-Batch SGD Works During Training

A typical mini-batch SGD training procedure follows these steps.

**Step 1: Initialize the model parameters.**

Start with initial weights and biases.

**Step 2: Shuffle the training dataset.**

Randomize the order of training observations so that mini-batches contain different combinations of data across epochs.

**Step 3: Divide the dataset into mini-batches.**

For example, divide 1,000 observations into 10 mini-batches, each containing 100 observations.

**Step 4: Perform a forward pass.**

For the selected mini-batch, calculate the model's predictions and the corresponding loss.

**Step 5: Calculate the gradients.**

Use backpropagation to calculate the gradients of the mini-batch loss with respect to the model parameters.

**Step 6: Update the model parameters.**

Use the gradient estimate to adjust the weights:

$$
w\leftarrow w-\alpha\nabla g_{\mathcal B}(w)
$$

**Step 7: Process the next mini-batch.**

Repeat the forward pass, gradient calculation, and parameter update for the remaining mini-batches.

**Step 8: Complete the epoch.**

Once all mini-batches have been processed, one epoch is complete.

The procedure is repeated for additional epochs until the stopping criterion is reached.

A common training workflow therefore follows:

**Shuffle data → Create mini-batches → Forward pass → Calculate loss → Backpropagation → Update parameters → Next mini-batch → Next epoch**


## 7. Why Approximate Gradients Can Be Useful

Mini-Batch SGD does not calculate the exact full-dataset gradient at every step.

Instead, it calculates an **estimate** using the selected mini-batch.

Because different mini-batches contain different observations, their gradients may point in slightly different directions.

This introduces variability, often called *gradient noise*, into the training process.

### Batch Gradient Descent

Batch Gradient Descent uses the full training gradient at every step.

For a fixed model and dataset, this produces a consistent update direction at a given parameter value.

Its trajectory is generally less noisy than that of SGD.

However, it may still encounter difficult optimization regions, including flat areas, saddle points, and local minima.

### Benefits and trade-offs

Mini-batches reduce the work needed before each update and can use parallel hardware efficiently. Their gradient estimates vary, so training may require careful learning-rate selection. Gradient noise can help exploration in nonconvex problems, but does not guarantee escape from a poor region or faster convergence.

If the dataset size is not divisible by batch size, the final batch may be smaller. Keeping that batch gives $\lceil n/m\rceil$ updates per epoch; discarding it gives $\lfloor n/m\rfloor$.

**Question 1:** With 1,003 observations and batch size 100, how many updates occur per epoch if the final batch is kept?

<details>
<summary>Show answer</summary>

There are 11 updates: ten batches of 100 and one batch of 3. The final batch gradient is averaged over its actual 3 observations.

</details>

**Question 2:** Does using a smaller batch necessarily make each parameter update smaller?

<details>
<summary>Show answer</summary>

No. Update size depends on the gradient and learning rate. A smaller batch changes the observations used to estimate the gradient; its gradient can have a larger or smaller magnitude than the full-data gradient.

</details>
