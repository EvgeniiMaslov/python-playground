

[**Working with data**](#Working-with-data)



1. Tabular data

* [Data imputation](#Data imputation)
* [Standardization and Normalization](#Standardization and Normalization)
* [Feature Selection](#Feature-selection)
* [Feature Extraction](#feature-extraction)
* [Feature importance](#Feature importance)
* [Strategies for Imbalance Data](#strategies-for-imbalance-data)
* [Dealing with categorical variables](#Dealing-with-categorical-variables)



3. Computer Vision

* [Contrast Normalization](#Contrast Normalization)







# Working with data



### Data imputation

1. One data imputation technique consists in replacing the missing value of a feature by an **average** value of this feature in the dataset. 

   **Плюсы:**

   - Просто и быстро.
   - Хорошо работает на небольших наборах численных данных.

   **Минусы:**

   - Значения вычисляются независимо для каждого столбца, так что корреляции между параметрами не учитываются.
   - Не работает с качественными переменными.
   - Метод не особенно точный.
   - Никак не оценивается погрешность импутации.

2. Another technique is to replace the missing value by a **value outside the normal range of values**. For example, if the normal range is [0, 1], then you can set the missing value to 2 or -1. The idea is that the learning algorithm will learn what is best to do when the feature has a value significantly different from regular values. 

3. Alternatively, you can replace the missing value by a **value in the middle of the range**. For example, if the range for a feature is [-1, 1], you can set the missing value to be equal to 0. Here, the idea is that the value in the middle of the range will not significantly affect the prediction. 

4. A more advanced technique is to use the missing value as the target variable for a **regression** problem. You can use all remaining features [x(1) i , x(2) i ,...,x(j≠1) i , x(j+1) i ,...,x(D) i ] to form a feature vector xˆi, set yˆi Ω x(j) , where j is the feature with a missing value. Then you build a regression model to predict yˆ from xˆ. Of course, to build training examples (xˆ, yˆ), you only use those examples from the original dataset, in which the value of feature j is present. 

5. Способ четвёртый: импутация данных с помощью **k-NN**

   Он использует сходство точек, чтобы предсказать недостающие значения на основании *k* ближайших точек, у которых это значение есть. Иными словами, выбирается *k* точек, которые больше всего похожи на рассматриваемую, и уже на их основании выбирается значение для пустой ячейки.

   
   
   Алгоритм сперва проводит импутацию простым средним, потом на основании получившегося набора данных строит дерево и использует его для поиска ближайших соседей. Взвешенное среднее их значений и вставляется в исходный набор данных вместо недостающего.
   
   **Плюсы:**
   
   - На некоторых датасетах может быть точнее среднего/медианы или константы.
   - Учитывает корреляцию между параметрами.

   **Минусы:**

   - Вычислительно дороже, так как требует держать весь набор данных в памяти.
- Важно понимать, какая метрика дистанции используется для поиска соседей. Имплементация в impyute [поддерживает только манхэттенскую и евклидову дистанцию](https://impyute.readthedocs.io/en/master/api/cross_sectional_imputation.html), так что анализ соотношений (скажем, количества входов на сайты людей разных возрастов) может потребовать предварительной нормализации.
   - Чувствителен к выбросам в данных (в отличие от SVM).
   
6. Finally, if you have a significantly large dataset and just a few features with missing values, you can increase the dimensionality of your feature vectors by adding a **binary indicator** feature for each feature with missing values. Let’s say feature j = 12 in your D-dimensional dataset has missing values. For each feature vector x, you then add the feature j = D + 1 which is equal to 1 if the value of feature 12 is present in x and 0 otherwise. The missing feature value then can be replaced by 0 or any number of your choice.



### Standardization and Normalization

Normalization is the process of converting an actual range of values which a numerical feature can take, into a standard range of values, typically in the interval [-1, 1] or [0, 1].
$$
\bar x^{(j)} = \frac{x^{(j)} - min^{(j)}}{max^{(j)} - min^{(j)}}
$$
Standardization (or z-score normalization) is the procedure during which the feature values are rescaled so that they have the properties of a standard normal distribution with µ = 0 and ‡ = 1, where µ is the mean (the average value of the feature, averaged over all examples in the dataset) and ‡ is the standard deviation from the mean.
$$
\hat x^{(j)} = \frac{x^{(j)} - \mu^{(j)}}{\sigma^{(j)}}
$$
Usually, if your dataset is not too big and you have time, you can try both and see which one performs better for your task. If you don’t have time to run multiple experiments, as a rule of thumb:

* unsupervised learning algorithms, in practice, more often benefit from standardization than from normalization; 
* standardization is also preferred for a feature if the values this feature takes are distributed close to a normal distribution (so-called bell curve); 
* again, standardization is preferred for a feature if it can sometimes have extremely high or low values (outliers); this is because normalization will “squeeze” the normal values into a very small range; 
* in all other cases, normalization is preferable.



### Feature Selection

A classic sequential feature selection algorithm is sequential backward selection (SBS), which aims to reduce the dimensionality of the initial feature subspace with a minimum decay in the performance of the classifier to improve upon computational efficiency.

The idea behind the SBS algorithm is quite simple: SBS sequentially removes features from the full feature subset until the new feature subspace contains the desired number of features. In order to determine which feature is to be removed at each stage, we need to define the criterion function, J, that we want to minimize. 

The criterion calculated by the criterion function can simply be the difference in performance of the classifier before and after the removal of a particular feature. Then, the feature to be removed at each stage can simply be defined as the feature that maximizes this criterion; or in more simple terms, at each stage we eliminate the feature that causes the least performance loss after removal. Based on the preceding definition of SBS, we can outline the algorithm in four simple steps: 

1. Initialize the algorithm with k = d, where d is the dimensionality of the full feature space, 𝑿𝑑. 
2. Determine the feature, 𝒙−, that maximizes the criterion: 𝒙− = argmax 𝐽(𝑿𝑘 − 𝒙), where 𝒙 ∈ 𝑿𝑘. 
3. Remove the feature, 𝒙−, from the feature set: 𝑿𝑘−1 = 𝑿𝑘 − 𝒙− ; 𝑘 = 𝑘 − 1. 
4. Terminate if k equals the number of desired features; otherwise, go to step 2

Also we can select features use random forest importance, l1 regularization.

### Feature Extraction

For the feature extraction you can use:

* Principal component analysis (PCA) for unsupervised data compression 
* Linear discriminant analysis (LDA) as a supervised dimensionality reduction technique for maximizing class separability 
* Nonlinear dimensionality reduction via kernel principal component analysis (KPCA)

### Feature importance

There are two ways to measure variable importance: 

1. By the decrease in accuracy of the model if the values of a variable are randomly permuted (type=1). Randomly permuting the values has the effect of removing all predictive power for that variable. The accuracy is computed from the out-of-bag data (so this measure is effectively a cross-validated estimate). 
2. By the mean decrease in the Gini impurity score (see “Measuring Homogeneity or Impurity”) for all of the nodes that were split on a variable (type=2). This measures how much improvement to the purity of the nodes that variable contributes. This measure is based on the training set, and therefore less reliable than a measure calculated on out-of-bag data.

Ensemble models improve model accuracy by combining the results from many models. Bagging is a particular type of ensemble model based on fitting many models to bootstrapped samples of the data and averaging the models. Random forest is a special type of bagging applied to decision trees. In addition to resampling the data, the random forest algorithm samples the predictor variables when splitting the trees. A useful output from the random forest is a measure of variable importance that ranks the predictors in terms of their contribution to model accuracy. The random forest has a set of hyperparameters that should be tuned using cross-validation to avoid overfitting.



### Strategies for Imbalance Data

* Undersample - Use fewer of the prevalent class records in the classification model.

* Oversample - Use more of the rare class records in the classification model, bootstrapping if necessary.

* Up weight or down weight - Attach more (or less) weight to the rare (or prevalent) class in the model.

* Data generation - Like bootstrapping, except each new bootstrapped record is slightly different from its source.

* Z-score - The value that results after standardization.

  

A variation of upsampling via bootstrapping (see “Undersampling”) is data generation by perturbing existing records to create new records. The intuition behind this idea is that since we only observe a limited set of instances, the algorithm doesn’t have a rich set of information to build classification “rules.” By creating new records that are similar but not identical to existing records, the algorithm has a chance to learn a more robust set of rules.

Highly imbalanced data (i.e., where the interesting outcomes, the 1s, are rare) are problematic for classification algorithms. One strategy is to balance the training data via undersampling the abundant case (or oversampling the rare case). If using all the 1s still leaves you with too few 1s, you can bootstrap the rare cases, or use SMOTE to create synthetic data similar to existing rare cases. Imbalanced data usually indicates that correctly classifying one class (the 1s) has higher value, and that value ratio should be built into the assessment metric.



One way to deal with imbalanced class proportions during model fitting is to assign a larger penalty to wrong predictions on the minority class. Via scikit-learn, adjusting such a penalty is as convenient as setting the class_weight parameter to class_weight='balanced', which is implemented for most classifiers.

Other popular strategies for dealing with class imbalance include upsampling the minority class, downsampling the majority class, and the generation of synthetic training examples.

```python
from sklearn.utils import resample
X_upsampled, y_upsampled = resample(
									X_imb[y_imb == 1],
									y_imb[y_imb == 1],
									replace=True,
									n_samples=X_imb[y_imb == 0].shape[0],
									random_state=123)
```

Similarly, we could downsample the majority class by removing training examples from the dataset. To perform downsampling using the resample function, we could simply swap the class 1 label with class 0 in the previous code example and vice versa.

There are two popular algorithms that oversample the minority class by creating synthetic examples: the **synthetic minority oversampling technique** (SMOTE) and the **adaptive synthetic sampling method** (ADASYN).

SMOTE and ADASYN work similarly in many ways. For a given example xi of the minority class, they pick k nearest neighbors of this example (let’s denote this set of k examples Sk) and then create a synthetic example xnew as xi + (xzi - xi), where xzi is an example of the minority class chosen randomly from Sk. The interpolation hyperparameter is a random number in the range [0, 1]. Both SMOTE and ADASYN randomly pick all possible xi in the dataset. In ADASYN, the number of synthetic examples generated for each xi is proportional to the number of examples in Sk which are not from the minority class. Therefore, more synthetic examples are generated in the area where the examples of the minority class are rare.



### Dealing with categorical variables

* Dummy variables - Binary 0-1 variables derived by recording factor data for use in regression and other models.
* Reference coding - The most common type of coding used by statisticians, in which one level of a factor is used as a reference and other factors are compared to that level
* One hot encoder - A common type of coding used in machine learning community in which all factors levels are retained. While useful for a certain machine learning algorithms, this approach is not appropriate for multiple linear regression.
* Deviation coding - A type of coding that compares each level against the overall mean as opposed to the reference level.

Factor variables need to be converted into numeric variables for use in a regression. The most common method to encode a factor variable with P distinct values is to represent them using P-1 dummy variables. A factor variable with many levels, even in very big data sets, may need to be consolidated into a variable with fewer levels. Some factors have levels that are ordered and can be represented as a single numeric variable.

Because of correlation between predictors, care must be taken in the interpretation of the coefficients in multiple linear regression. Multicollinearity can cause numerical instability in fitting the regression equation. A confounding variable is an important predictor that is omitted from a model and can lead to a regression equation with spurious relationships. An interaction term between two variables is needed if the relationship between the variables and the response is interdependent.









### Contrast Normalization



One of the most obvious sources of variation that can be safely removed for many tasks is the amount of contrast in the image. Contrast simply refers to the magnitude of the difference between the bright and the dark pixels in an image.

Global contrast normalization (GCN) aims to prevent images from having varying amounts of contrast by subtracting the mean from each image, then rescaling it so that the standard deviation across its pixels is equal to some constant s. This approach is complicated by the fact that no scaling factor can change the contrast of a zero-contrast image (one whose pixels all have equal intensity). Images with very low but non-zero contrast often have little information content. Dividing by the true standard deviation usually accomplishes nothing more than amplifying sensor noise or compression artifacts in such cases.

