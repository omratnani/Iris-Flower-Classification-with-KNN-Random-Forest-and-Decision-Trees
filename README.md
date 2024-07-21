# Iris Flower Classification with KNN, Random Forest, and Decision Trees

## Overview

This project demonstrates the classification of Iris flowers into three species: Setosa, Versicolor, and Virginica. Using the classic Iris dataset, we implemented three different machine learning algorithms: K-Nearest Neighbors (KNN), Decision Trees, and Random Forest. This project aims to compare the performance of these algorithms in terms of accuracy and provide insights into their respective strengths.

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris) is a well-known dataset in the machine learning community, containing 150 observations of iris flowers. Each observation has four features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

There are three classes in the dataset:
- Setosa
- Versicolor
- Virginica

## Project Structure

1. **Data Preparation and Visualization**:
   - Load the Iris dataset using `sklearn.datasets.load_iris`.
   - Convert the dataset into a Pandas DataFrame for easier manipulation.
   - Visualize the data using Matplotlib to understand the distribution of different species based on the features.

2. **K-Nearest Neighbors (KNN)**:
   - Split the data into training and testing sets.
   - Train a KNN classifier with 3 neighbors.
   - Evaluate the model using accuracy score and confusion matrix.
   - Visualize the confusion matrix with a heatmap.

3. **Decision Trees**:
   - Split the data into training and testing sets.
   - Train a Decision Tree classifier.
   - Evaluate the model using accuracy score.
   - Visualize the decision tree to understand the decision-making process.

4. **Random Forest**:
   - Split the data into training and testing sets.
   - Train a Random Forest classifier with 150 trees.
   - Evaluate the model using accuracy score.

## Installation

To run this project, you need to have Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


## Results

- **K-Nearest Neighbors (KNN)**:
  - Accuracy: 1.00
  - Confusion Matrix:

  ![KNN Confusion Matrix](images/knn_confusion_matrix.png)

- **Decision Trees**:
  - Accuracy: 1.00
  - Decision Tree Visualization:

  ![Decision Tree](images/decision_tree.png)

- **Random Forest**:
  - Accuracy: 1.00

## Conclusion

This project demonstrates how different machine learning algorithms can be applied to classify iris flowers and highlights their performance in terms of accuracy. KNN, Decision Trees, and Random Forest all achieved high accuracy on this dataset. The decision tree visualization provides insight into the model's decision-making process, while the Random Forest model benefits from the aggregation of multiple decision trees to improve accuracy and robustness.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The Iris dataset used in this project is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris).
