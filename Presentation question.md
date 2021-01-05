#Mid Question list
--
###1. **Negative $R^2$**

The main reason is we are doing on the test set using the fitted model from train set. So it is not a normal $R^2$ from regression model.

$R^2=1 - \frac{SS_{Res}}{SS_{T}}$. 
When the sum square errors of the model $(SS_{Res})$ is larger than the total sum square  $(SS_{T})$, then it becomes negative. 


$R^2$ is not always the square of anything, so it can be negative. $R^2$ is negative only when the chosen model does not follow the trend of the data, so fits worse than a horizontal line.
A negative $R^2$ means the model is terrible.

###2. **Overfitting Problem**
A key challenge with overfitting, and with machine learning in general, is that we can’t know how well our model will perform on new data until we actually test it.
To address this, we can split our initial dataset into separate training and test subsets.

*Slove it*

* Cross-validation
Use your initial training data to generate multiple mini train-test splits. Use these splits to tune your model.
* Train with more data
* Remove features
* Ensembling


###3. **Multicollinearity**
Since we focus on the prediction, multicollinearity is not a big issue in the linear regression method.

*The variables need to be independent!*

Yes, this may be a problem. But if we directly drop these variable, then it will loss information. So the better way is to find some better models.

###4. **Decision Tree Information Gain**
Information gain is a decrease in entropy. It computes the difference between entropy before split and average entropy after split of the dataset based on given attribute values.
In orther word, it means it may have the good condition on each node.

For an easy example, what contributes student scores?


###5. **RF Reducing Variance**
Increasing the number of trees will reduce the variance of the estimator. This is an obvious consequence of one of the CLTs $$\sigma_{\bar{X}}=\frac{\sigma}{\sqrt{n}}$$ Because each tree is a binomial trial, and the prediction of the forest is the average of many binomial trials. 

This can make the predictions have less volatility because the trees only have to explain groups of the data, instead of each observation. So lots of trials will cut the standard error of the mean in half.

###6. **GB Reducing Bias**

The reason why reducing the bias is that we want our model to fit with our data better as long as it does not overfit. 
High bias means that our model didn’t learn enough from the data. The gradient boosting combines of many weak learners like(*Regression*,*Random Forest*, as well as the *Decision Tree*). So it 





###7. **MAE/RMSE/MAPE**
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)


