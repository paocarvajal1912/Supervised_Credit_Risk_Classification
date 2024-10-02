# Supervised Credit Risk Classification
# Summary Report

## Overview of the Analysis

The goal of this analysis is to develop a robust model capable of predicting the creditworthiness of borrowers from a peer-to-peer lending platform. The model is designed to accurately differentiate between healthy loans and high-risk loans, allowing the company to minimize losses from defaulted loans while maximizing profitability by approving creditworthy borrowers.

The data used in this analysis consists of historical loan records and key financial indicators that reflect the borrower's financial standing and loan attributes. These features include loan size, interest rate, borrower income, debt-to-income ratio, number of credit accounts, total debt, and the presence of derogatory marks. This information provides a comprehensive view of a borrower's ability to repay the loan.

The dataset includes 77,536 loan records, of which 75,036 are categorized as healthy loans, while 2,500 represent defaulted (high-risk) loans. As expected, the data is imbalanced, with a small proportion of high-risk loans, which mirrors real-world scenarios where the majority of borrowers meet their financial obligations. Advanced machine learning techniques and strategies to handle imbalanced data have been applied to mitigate this limitation and ensure the model performs optimally in identifying both high-risk and healthy loans.


## Stages of the Machine Learning Process

The process involved in machine learning modelling consist of the following steps:

1. **Collection and preparation of the data.** This was done prior to this analysis, and the data is available this [csv file]("Resources/lending_data.csv"). An overview of the initial dataset is shown below.

![DataOverview](Images/01_DataOverview.png)

2. **Definition of training and test data.** We split the data into **training** and **testing** data sets, by applying the [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function from the Model Selection module of the SciKit-learn library in Python. The **training data** is used to fit the model, and the **test data** (25% of the total set) is used to evaluate the model's predictive performance. 

3. **Creation of potencial models for evaluation.** We create two Logistic Regression models using the [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) method from the Scikit-learn's Linear Model module:
    - **Machine Learning Model 1**: this model directly uses the original loan data. 
    - **Machine Learning Model 2**: this model improves upon Model 1 by applying **oversampling** to address the imbalance between high-risk and healthy loans in the dataset. We used the [`RandomOverSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler) method, from the `Imblearn` library, by randomly select defaulted loans (with replacement), to equalize the number of observations in both classes. The resulting class distribution is shown below.

<img src="Images/03_ResampleNumbers.png" width="250" />

4. **Fitting of the model.** We fit the **training data** to each model using the [`fit` method](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=fit#sklearn.linear_model.LogisticRegression.fit) of the Logistic Regression model. The optimization solver used is the Limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS). This optimizer does not require feature scaling. The model also incorporates **regularization** to prevent overfitting to the training data.

5. **Predicting the quality of the loans** (healthy or high-risk) on the *test sample data* for both models. We used the [`predict` method](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=fit#sklearn.linear_model.LogisticRegression.predict) of the LogisticRegression model to generate this predictions. 

6. **Comparison of results.** The following key metrics were used to evaluate model performance:
    - [balance accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
    - [precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
    - [recall score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html). 

    > The **precision score** measures how accurately the model predicts a specific class. For example, out of loans predicted to be high-risk, what percentage were actually high-risk loans. 

    > The **recall score** imeasures how well the model identifies a particular class. For instance, out of all actual high-risk loans, what percentage the model correctly classifies as high-risk. 

    > The **balance accuracy score** accounts for class imbalance ans it is calculated as the average of the recall scores for both the healthy loan class and the high-risk loan class.



## Results

Here are the classification reports of both models.

<img src="Images/02_UmbalancedClassifiacationReport.png" width="700" />
<img src="Images/04_RebalancedClassifiacationReport.png" width="700" />

The key metrics described earlier - **balanced accuracy**, **precision**, and **recall**— are summarized for each model below.

### **Machine Learning Model 1 metrics:**
- **Balance Accuracy score:** 95.8%
- **Precision score:** 
    - Healthy loans: 100%
    - High-risk loans: 85%
    - Average: 99%
- **Recall score:**
    - Healthy loans: 99%
    - High-risk loans: 92%
    - Average:99%

### **Machine Learning Model 2 metrics:**
- **Balance Accuracy score:** 99.6%
- **Precision score: **
    - Healthy loans: 100%
    - High-risk loans: 85%
    - Average: 100%
- **Recall scores:**
    - Healthy loans: 99% 
    - High-risk loans: 100%
    - Average: 99%


  
### Key Conclusions:

* The **Balance Accuracy** shows a significant improvement from 95.8% in Model 1 to 99.6% in Model 2 when applying oversampling. This improvement is primarily due to san increase in the recall for hig-risk loans, as detailed below.

* With Oversampling, the recall for high-risk loans increased from **92%** to **100%**, meaning all previously misclassified high-risk-loans (false negatives) now are now correctly identified. This is crucial for managing the costs associated with loans defaults. Without resampling, the bank would have faced potential losses from hiugh-risk loans incorrectly classified as healthy. By addressing the class imbalance with oversampling, the model now prevents such default-related cost at the highest level.

* The precision for high-risk-loans remains at **85%** in both models. This indicates that the oversampling did not enhanced the model's ability to more accurately identigy high-risk loans. As a result, the proportion of healthy loans incorrectly classified as high-risk (false positives) remains unchanged, and the associated opportunity cost from rejecting these loans persists in both models.

* **Healthy loans** are predicted with a precision of 100% and a recall of 99% in both models, rflecting near perfect performance. Since oversampling does not affect this class, it might not be necessary for healthy loans, given the already high classification accuracy.


## Summary

In this analysis, we developed and evaluated two Logistic Regression models to predict loan quality. **Model 1** was trained on the original, imbalanced dataset, while **Model 2** incorporated oversampling techniques to address the class imbalance.

### Key Findings

- **Overall Performance:**
  - Both models achieved **balanced accuracy scores above 95%**, demonstrating strong overall predictive capabilities.
  - Both models exhibited **near-perfect metrics** for identifying healthy loans, with **precision and recall rates approaching 100%**.

- **High-Risk Loan Prediction:**
  - **Model 2** significantly outperformed **Model 1** in detecting high-risk loans, achieving a **recall of 100%** compared to **92%** for Model 1. This enhancement ensures that all default-prone loans are accurately identified, thereby minimizing the financial risks associated with loan defaults.

- **Precision in High-Risk Classification:**
  - Both models exhibited an **85% precision** in predicting high-risk loans. This indicates that **15% of loans classified as high-risk were actually healthy**, resulting in an opportunity cost due to the rejection of profitable loans.

### Recommendations

Based on the performance metrics, we **strongly recommend adopting Model 2** for the following reasons:

1. **Enhanced Risk Mitigation:**
   - **Model 2**’s ability to achieve a **100% recall** for high-risk loans ensures that all potential defaults are identified, effectively safeguarding the company against significant financial losses.

2. **Balanced Performance:**
   - Despite the **15% opportunity cost** from misclassifying healthy loans as high-risk, the **85% precision** remains robust. This trade-off is justified by the substantial reduction in default-related costs, which pose a far greater threat to the company's financial stability.

3. **Strategic Advantage:**
   - The superior recall of **Model 2** provides a critical edge in managing credit risk, enabling the company to make informed lending decisions that prioritize long-term profitability and sustainability.

### Conclusion

The deployment of **Model 2** offers a compelling balance between identifying high-risk loans and maintaining a high precision in classification. By prioritizing the prevention of loan defaults, Model 2 delivers unparalleled performance in mitigating financial risks, making it an indispensable tool for the company's credit risk management strategy. While the opportunity cost of misclassifying some healthy loans exists, the overarching benefit of eliminating default-related losses outweighs this drawback.


--------------------------
## Technologies Used

This project is developed in **Python** using **JupyterLab** as the interactive environment. The analysis relies heavily on the `Scikit-learn (SkLearn)` library, particularly the following modules:
- `metrics` for performance evaluation
- `preprocessing` for data preparation
- `model_selection` for splitting data and cross-validation
- `linear_model` for model development

Additional libraries used include:
- `NumPy` for numerical computations
- `Pandas` for data manipulation and analysis
- `Pathlib` for file handling
- `Warnings` to manage warning messages in the code

## Installation Guide

The project is provided as a Jupyter notebook. If you do not have Jupyter installed, you can follow the installation instructions [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

Once Jupyter is installed, you can run the notebook by launching JupyterLab or Jupyter Notebook from your terminal.

## Contributors

This project was created by **Paola Carvajal Almeida**.

Feel free to reach out via email: [paola.antonieta@gmail.com](mailto:paola.antonieta@gmail.com)

You can also view my LinkedIn profile [here](https://www.linkedin.com/in/paolacarvajal/).

## License

This project is licensed under the **MIT License**. This license permits the use, modification, and distribution of the code, provided that the original copyright and license notice are retained in derivative works. The license does not include a patent grant and absolves the author of any liability arising from the use of the code.

For more details, you can review the full license text in the project repository.







