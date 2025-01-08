# Getting Started


`obliquetree` combines advanced capabilities with efficient performance. It supports **oblique splits**, leveraging **L-BFGS optimization** to determine the best linear weights for splits, ensuring both speed and accuracy. The library also handles **categorical features**, making it suitable for datasets with mixed data types. Even when used in its **traditional mode**, without oblique splits, `obliquetree` outperforms `scikit-learn` in terms of speed and adds support for categorical variables, providing a significant advantage over many traditional decision tree implementations.

When the **oblique feature** is enabled, `obliquetree` dynamically selects the optimal split type between oblique and traditional splits. If no weights can be found to reduce impurity, it defaults to an **axis-aligned split**, ensuring robustness and adaptability in various scenarios.

In very large trees (e.g., depth 20 or more), the performance of `obliquetree` may converge closely with **traditional trees**. The true strength of `obliquetree` lies in their ability to perform exceptionally well at **shallower depths**, offering improved generalization with fewer splits. Moreover, thanks to linear projections, `obliquetree` significantly outperform traditional trees when working with datasets that exhibit **linear relationships**.

```{note}
1. **Data Format**:
   - `obliquetree` expects input data in **Fortran order**. If the data is not in Fortran order, the library will automatically create a copy in the correct format.

2. **Splitting Criteria**:
   - For **regression tasks**, the library currently uses **Mean Squared Error (MSE)**.
   - For **classification tasks**, it uses **Gini Impurity**.
   - Future versions will include additional splitting criteria.

3. **Data Standardization**:
   - It is **highly recommended** to standardize your data before using `obliquetree` for better performance and stability.

4. **Flexibility**:
   - The library can be used as a **traditional decision tree** without oblique splits. In this mode, it outperforms scikit-learn in speed and supports categorical variables.

5. **Handling Missing and Infinite Values**:
   - For **traditional axis-aligned splits**, `obliquetree` can handle `NaN` and `Inf` values.
   - For **oblique splits**, any `NaN` or `Inf` values must be **imputed** before use.
```

---

## Parameter Descriptions

### General Parameters
- **`use_oblique` (bool, default=True):**
  - Specifies whether to use oblique splits.
  - When set to `True`, the decision tree can use both linear combinations of features and axis-aligned to make splits.
  - When `False`, the tree uses traditional axis-aligned splits.

- **`max_depth` (int, default=-1):**
  - Maximum depth of the tree.
  - If set to `-1`, the tree expands until all leaves are pure or contain fewer than `min_samples_split` samples.

- **`min_samples_leaf` (int, default=1):**
  - The minimum number of samples required to be at a leaf node.

- **`min_samples_split` (int, default=2):**
  - The minimum number of samples required to split an internal node.

- **`min_impurity_decrease` (float, default=0.0):**
  - A node is split if the impurity decrease is greater than or equal to this value.

- **`ccp_alpha` (float, default=0.0):**
  - Complexity parameter for Minimal Cost-Complexity Pruning.
  - Larger values result in more aggressive pruning.

### Oblique-Specific Parameters
- **`n_pair` (int, default=2):**
  - The number of features to consider for oblique splits.
  - All possible combinations of `n_pair` features from the continuous feature set are tested.
  - **Example:** If there are 20 continuous features and `n_pair=3`, the algorithm will evaluate $\binom{20}{3}$ combinations.
  - **Recommendation:** Use small values like 2, 3, or the total number of continuous features minus a small number. Large values can lead to computational challenges due to the combinatorial explosion.

- **`gamma` (float, default=1.0):**
  - Controls the separation strength in oblique splits.
  - Higher values enforce stronger separation in the loss function.

- **`max_iter` (int, default=100):**
  - Maximum number of iterations for the L-BFGS optimization algorithm.

- **`relative_change` (float, default=0.001):**
  - Early stopping threshold for L-BFGS optimization.
  - Smaller values lead to longer optimization times but can improve split quality.

- **`random_state` (int, default=None):**
  - Seed for random number generation in oblique splits.
  - Ensures reproducibility of results when set.

### Categorical Data Support
- **`categories` (List[int], default=None):**
  - A list of column indices representing categorical features.
  - Categorical features are not used directly in oblique splits but are fully supported in axis-aligned splits.

---


```{important}
The `n_pair` parameter is critical for oblique splits. It defines how many features are combined to evaluate split candidates. For example:
- If there are **20 continuous features** and `n_pair=3`, the algorithm evaluates $\binom{20}{3} = 1140$ combinations.
- This combinatorial nature makes large values computationally expensive or infeasible.

**Recommended Values:**
- `n_pair=2` or `n_pair=3` for most use cases.
- `n_pair` equal to the number of continuous features or slightly less for maximum flexibility.

Avoid setting `n_pair` to produce high number of combinations, as the computational cost grows rapidly and can make the algorithm impractical for large feature sets.
```