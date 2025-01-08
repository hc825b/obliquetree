# Categorical Support in Decision Trees

In decision tree algorithms, handling categorical features requires special consideration. Unlike numerical features where natural ordering exists, categorical variables need different strategies for finding optimal splits. Here's how we handle categorical splits for different types of problems:

1. **Binary Classification and Regression:**  
   - We group categories based on their relationship with the target variable
   - Each category's contribution to the target is measured through mean target values
   - Categories are sorted and systematically grouped to find the most effective split point
   
2. **Multiclass Classification:**  
   - We examine how each category relates to multiple classes
   - Since classes don't have natural ordering
   - We explore different ways to group categories by considering all possible class orderings

---

## Binary Classification and Regression

For each categorical value $k$, we calculate its mean target value to understand how that category relates to what we're trying to predict:

$$
\mu_k = \frac{\sum_{i \in C_k} y_i w_i}{\sum_{i \in C_k} w_i}
$$

Where:
- $C_k$: Set of samples belonging to category $k$
- $y_i$: Target value for sample $i$
- $w_i$: Sample weight for instance $i$

### Detailed Steps:
1. **Sort categories** by their mean target values ($\mu_k$) in descending order:

   $$
   \mu_1 \geq \mu_2 \geq \dots \geq \mu_m
   $$

2. **Evaluate all possible splits:** 
For each potential split point $t$ between 1 and $m-1$ categories:
   - Left node: Top $t$ categories (higher mean values)
   - Right node: Remaining $m-t$ categories (lower mean values)

3. Select the split that gives the best prediction performance by minimizing:
   - For regression: Mean Squared Error 
   - For binary classification: Gini Impurity

---

## Multiclass Classification

When dealing with multiple classes (3+ categories), we need a more sophisticated approach since there's no natural ordering between classes.

### 1. **Class Distribution Calculation**

Calculate how each category relates to each possible class:

$$
p_{k,j} = \frac{\sum_{i \in C_k} I(y_i=j)w_i}{\sum_{i \in C_k} w_i}
$$

Where:
- $p_{k,j}$: Proportion of class $j$ in category $k$
- $I(y_i=j)$: Equals 1 if sample $i$ belongs to class $j$, 0 otherwise

### 2. **Permutation-based Ordering**

Since we don't know the best way to order classes, we try all possible arrangements:
- Consider all ways to order the classes
- For each ordering $\pi$, we compare categories $k_1$ and $k_2$ based on:
  - Their class distributions matching up to some position
  - The first position where they differ determines their order

### 3. **Split Evaluation**

For each class ordering $\pi$ and split point $t$:
- Calculate node purity using Gini impurity:

  $$
  Gini(S) = 1 - \sum_{j=1}^C p_j^2
  $$

  Where $p_j$ is the proportion of class $j$ in a node
  
- Calculate weighted average impurity for both child nodes:

  $$
  Gini_{split} = \frac{n_L}{n}Gini(S_L) + \frac{n_R}{n}Gini(S_R)
  $$

  Where:
  - $n_L, n_R$: Number of samples (weighted) in left and right nodes
  - $S_L, S_R$: Samples in left and right nodes

### 4. **Best Split Selection**

Find the optimal split by minimizing impurity across all possible:
- Class orderings ($\pi$)
- Split points ($t$)

$$
split^* = \arg\min_{\pi, t} Gini_{split}(\pi, t)
$$

This comprehensive approach ensures we find the most effective way to group categorical values, regardless of the problem type or number of classes.