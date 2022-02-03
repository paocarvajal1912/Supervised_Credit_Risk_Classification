# Supervised Credit Risk Classification
# Summary Report

## Overview of the Analysis

The purpose of this anaylsis is to construct a model to predict the creditworthness of borrowers from a peer-to-peer lending service company. The model should be able to distinguish between healthy and highly-risk loans so the company minimize the cost of approving highly-risky loans that default, and maximize the profit associated to approve healthy loans.

The data used in the model corresponds to historical loans and key financial variables associated to the borrower financial position and the loan characteristics. This data helps to determine whether a person will be able to pay back a loan or not. These variables are the loan size, the interest rate, the borrower income, the borrower debt to income ratio, the number of accounts of the borrower, his/her total debt, and whether he/she has any derogatory marks.

The data consist of 77,536 loans, from which 75,036 are healthy loans, and 2,500 are defaulted loans (high-risk loans). We can see that there is a considerable imbalance in the data, which is usual given that people tend to pay their obligations. We have applied machine learning and methodologies that allow us to deal with the imbalance limitation in the data.


## Stages of the Machine Learning Process

The process involved in machine learning modelling consist on the following steps:

1. **Collection and preparation of the data.** This was made previously to this analysis, and the data was provided in a csv file. An overview of the initial dataset follows.

![DataOverview](Images/01_DataOverview.png)

2. **Definition of training and test data.** For this, we split the data into **training** and **testing** data sets, by applying the [`train_test_split function`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from the Model Selection module of the SKlearn library in Python. The **training data** is used to fit the model, and the **test data** (25% of the set) is used to measure the quality of the predictions. 

3. **Creation of potencial models to evaluate.** We create two Logistic Regression machine learning models by applying the [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) method from the Linear Model module of SKLearn. The first one, **Machine Learning Model 1**, directly uses the original loan data. The second one, **Machine Learning Model 2**, attempts to improve the model by applying a technique called **oversampling** to deal with the imbalance between the amount of high-risk and healthy loans in the original data. The [`RandomOverSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler) method, available in the Over Sampling module of the Imblearn library, allow us to artificially increase the amount of defaulted loans in the data by randomly select defaulted loans with reposition, allowing to equalize the number of samples in both classes. An screenshot of the number of observations after the equalization follows.

<img src="Images/03_ResampleNumbers.png" width="250" />

4. **Fitting of the model.** We fit the **training data** to the model, which means to adjust the parameters for the best possible replication of the targets included in the data. The [`fit` method](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=fit#sklearn.linear_model.LogisticRegression.fit) of the Logistic Regression model is used for that purpose. The solver function used for the optimization of the fit is the Limited-memory Broyden–Fletcher–Goldfarb–Shanno. The use of this optimizer does not require the scalation of the features before applying the algorithm. Also, the fit method applied include **regularization**, which is the inclusion of adjustments to prevent overfitting to the training data.

5. **Predict the quality of the loans** (healthy or high-risk) of the separated *test sample data* in both models. For this, we use the [`predict` method](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=fit#sklearn.linear_model.LogisticRegression.predict) of the LogisticRegression model. 

6. **Compare results.** For this we use three key metrics, which are the [balance accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html), the [precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html), and the [recall score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html). 

>The **precision score** measures how good is the model in predicting one particular class. For example, of all high-risk loans predictions, which percentage were actually high-risk loans. 

>The **recall score** indicates how good is the model in recognizing one particular class. For example, of all the high-risk loans, what percentage the model correctly classifies as high-risk. 

>The **balance accuracy score** is a measurement design to account for the imbalance in the model, and it is defined as the average between the recalls of both classes. This means, the average between the recall of the healthy loan class, and the recall of the high-risk loan class.



## Results

Here are the classification reports of both models.

<img src="Images/02_UmbalancedClassifiacationReport.png" width="700" />
<img src="Images/04_RebalancedClassifiacationReport.png" width="700" />

The results of the three key metrics described in the last paragraph are shown below for each model.

* **Machine Learning Model 1 metrics:**
  * Balance Accuracy score: 95.8%
  * Precision score: 
      * Healthy loans: 100%
      * High-risk loans: 85%
      * Average: 99%
  * Recall scores:
      * Healthy loans: 99%
      * High-risk loans: 92%
      * Average:99%
     



* **Machine Learning Model 2 metrics:**
  * Balance Accuracy score: 99.6%
  * Precision score: 
      * Healthy loans: 100%
      * High-risk loans: 85%
      * Average: 100%
  * Recall scores:
      * Healthy loans: 99% 
      * High-risk loans: 100%
      * Average: 99%


  
From this results we can make the following conclusions:

* The Balance Accuracy shows that there is an overall significant improvement in the model recall from 95.8% to 99.6% when applying oversampling. This improvement comes from the improvement in the recall of hig-risk loans, as it will be explained next.

* When including resampling, there is an optimal improvement in the recall from 92% to 100%. That means that all of the negative falses (loans declared as healthy when they were highly risky) now are correctly labeled as 1 (high-risk loan). This is great in terms of controling the cost associated to the default of loans. Without resampling, the bank would have to absorve the cost of the high-risk loans that were incorrectly classified as healthy. Fortunately, when applying the resample to control the imbalance, we see that the cost of default is prevented by the improved model at the highest level.

* High-risk loans has the same precision of 85% in both models. This means that the oversampling does not increase the capacity of the model to predict more efficiently the high-risk loans. In other words, the proportion of incorrect predictions of healthy loans as high-risk loans (false positives), and the opportunity cost associated to that wrong assesment is the same in both models.

* Healthy loans are predicted with a precision of 100%, which is optimal, and a recall of 99%, which is also very high. This happens in both models. Oversampling does not make a difference in the classification of this class, and it may be arguably consider not necesary either due to the high performance.


## Summary

Two Logistic Regression machine learning models were constructed and compared on predicting the quality of loans. Model 1 used the imbalanced oriinal data, and Model 2 included a treatment for that imbalance. 

Both models have positive results. Both have a balance accuracy above 95%, and almost perfect metrics for healthy loans recognition. That said, model 2 have better performance in recognizing high-risk loans, with a recall of 100%, above the 92% of Model 1.

The weaker metric on both was an 85% precision in the prediction of high-risk loans. This implies the opportunity cost of rejecting healthy loans due to a wrong classification as high-risk loans.

We recommend the use of Model 2, because of its excellent balanced performance. Its capacity to recognize a high-risk loan with an statistically perfect score will allow to avoid the more important and dangerous costs: the cost of default. We acknoledge the draw back of falsely classifying healthy loans as risky 15% of the time. However that opportunity cost does not imply a risk for the business survival, and still 85% of the time the classification is done correctly. The more important characteristics is to prevent defaults, and the performance of the model is statistically unbeatable in that aspect.

--------------------------
## Technologies used

This code is write in Python, in Jupyter lab.
The analys was made with an extensively use of the `*SkLearn*` library, and its `*metrics*`, `*preprocessing*`, `*model_selection*`, and `*linear_model*` modules.
Other libriries used are `NumPy`, `Pandas`, `Pathlib`, and `Wanrnings`.

## Instalation Guide
The file is a jupyter notebook. If you don't have jupyter notebook, you can install it following the instruction [here.](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)




## Contributors
This project was made by Paola Carvajal Almeida.

Contact email: paola.antonieta@gmail.com

See my LinkedIn profile [here.](https://www.linkedin.com/in/paolacarvajal/)


## License
This project uses a MIT license. This license allows you to use the licensed material at your discretion, as long as the original copyright and license are included in your work files. This license does not contain a patent grant,  and liberate the authors of any liability from the use of this code.






