# Saving and Loading a Model

The `obliquetree.Classifier` or  `obliquetree.Regressor` model can be saved and loaded using either Python's `pickle` module or the built-in `export_tree` and `load_tree` utilities provided by the `obliquetree` package.

**You can save the model as a binary file using `pickle` and load it later:**

```python
from obliquetree import Classifier
import pickle

# Initialize and train the model
clf = Classifier()
clf.fit(X_train, y_train)

# Save the model to a file
PICKLE_SAVE_PATH = "out.pickle"

with open(PICKLE_SAVE_PATH, "wb") as file:
    pickle.dump(clf, file)

# Load the model from the file
with open(PICKLE_SAVE_PATH, "rb") as file:
    saved_clf = pickle.load(file)
```

Alternatively, you can use `export_tree` and `load_tree` from the `obliquetree.utils` module to save the model as a JSON file or a dictionary.


**You can save the model to a JSON file and load it later:**

```python
from obliquetree.utils import export_tree, load_tree

# Export the trained model to a JSON file
export_tree(clf, out_file="out.json")

# Load the model from the JSON file
saved_clf = load_tree("out.json")
```

**You can also export the model to a Python dictionary and reload it:**

```python
from obliquetree.utils import export_tree, load_tree

# Export the trained model to a dictionary
tree_dict = export_tree(clf)

# Load the model from the dictionary
saved_clf = load_tree(tree_dict)
```