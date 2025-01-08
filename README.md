# obliquetree

`obliquetree` is an advanced decision tree implementation designed to provide high-performance and interpretable models. It supports both classification and regression tasks, enabling a wide range of applications. By offering traditional and oblique splits, it ensures flexibility and improved generalization with shallow trees. This makes it a powerful alternative to regular decision trees.


![Tree Visualization](docs/source/_static/tree_visual.png)

-----
## Installation 
To install `obliquetree`, use the following pip command:

```bash
pip install obliquetree
```
-----

## Documentation
For example usage, API details, comparisons with axis-aligned trees, and in-depth insights into the algorithmic foundation, we **strongly recommend** referring to the full [documentation](https://obliquetree.readthedocs.io/en/latest/).

---
## **Key Features**

- **Oblique Splits**  
  Perform oblique splits using linear combinations of features to capture complex patterns in data. Supports both linear and soft decision tree objectives for flexible and accurate modeling.

- **Axis-Aligned Splits**  
  Offers conventional (axis-aligned) splits, enabling users to leverage standard decision tree behavior for simplicity and interpretability.

- **Feature Constraints**  
  Limit the number of features used in oblique splits with the `n_pair` parameter, promoting simpler, more interpretable tree structures while retaining predictive power.

- **Seamless Categorical Feature Handling**  
  Natively supports categorical columns with minimal preprocessing. Only label encoding is required, removing the need for extensive data transformation.

- **Robust Handling of Missing Values**  
  Automatically assigns `NaN` values to the optimal leaf for axis-aligned splits.

- **Customizable Tree Structures**  
  The flexible API empowers users to design their own tree architectures easily.

- **Exact Equivalence with `scikit-learn`**  
  Guarantees results identical to `scikit-learn`'s decision trees when oblique and categorical splitting are disabled.

- **Optimized Performance**  
  Outperforms `scikit-learn` in terms of speed and efficiency when oblique and categorical splitting are disabled:
  - Up to **50% faster** for datasets with float columns.
  - Up to **200% faster** for datasets with integer columns.

  ![Performance Comparison (Float)](docs/source/_static/sklearn_perf/performance_comparison_float.png)

  ![Performance Comparison (Integer)](docs/source/_static/sklearn_perf/performance_comparison_int.png)


----
### Contributing
Contributions are welcome! If you'd like to improve `obliquetree` or suggest new features, feel free to fork the repository and submit a pull request.

-----
### License
`obliquetree` is released under the MIT License. See the LICENSE file for more details.