[**Data Representation Design Patterns**](#Data Representation Design Patterns)

* [Simple Data Representations](#Simple Data Representations)
  * [Numerical inputs](#Numerical inputs)
  * [Categorical inputs](#Categorical inputs)
* [Hashed Feature](#Hashed Feature)
* [Embeddings](#Embeddings)
* [Feature Cross](#Feature Cross)
* [Multimodal Input](#Multimodal Input)
* [Data imputation strategies](#Data imputation)
* [Feature Selection](#Feature-selection)
* [Feature importance](#Feature importance)



[**Problem Representation Design Patterns**](#Problem Representation Design Patterns)

* [Reframing](#Reframing)
* [Multilabel](#Multilabel)
* [Cascade](#Cascade)
* [Neutral Class](# Neutral Class)
* [Rebalancing](#Rebalancing)



[**Model Training Patterns**](#Model Training Patterns)

* [Useful Overfitting](#Useful Overfitting)
* [Checkpoints](#Checkpoints)
* [Transfer Learning](#Transfer Learning)
* [Distribution Strategy](#Distribution Strategy)



[**Design Patterns for Resilient Serving**](#Design Patterns for Resilient Serving)

* [Stateless Serving Function](#Stateless Serving Function)
* [Batch Serving](#Batch Serving)
* [Continued Model Evaluation](#Continued Model Evaluation)
* [Two-Phase Predictions](#Two-Phase Predictions)



[**Reproducibility Design Patterns**](#Reproducibility Design Patterns)

* [Transform](#Transform)
* [Repeatable Splitting](# Repeatable Splitting)
* [Bridged Schema](#Bridged Schema)
* [Windowed Inference](#Windowed Inference)
* [Workflow Pipeline](#Workflow Pipeline)
* [Feature Store](#Feature Store)
* [Model Versioning](#Model Versioning)



[**Responsible AI**](#Responsible AI)

* [Heuristic Benchmark](#Heuristic Benchmark)
* [Explainable Predictions](#Explainable Predictions)



[**Machine Learning techniques and tasks**](#Machine-Learning-techniques-and-tasks)

* [Active learning](#Active learning)
* [Semi-supervised learning](#Semi-supervised learning)
* [One-shot learning](#One-shot learning)
* [Zero-shot learning](#Zero-shot learning)
* [Learning to Rank](#Learning to Rank)
* [Learning to Recommend](#Learning to Recommend)
* [Прогнозирование временных рядов](#Прогнозирование временных рядов)







## Data Representation Design Patterns



At the heart of any machine learning model is a mathematical function that is defined to operate on specific types of data only. At the same time, real-world machine learning models need to operate on data that may not be directly pluggable into the mathematical function. 

### Simple Data Representations



#### Numerical inputs

Most modern, large-scale machine learning models (random forests, support vector machines, neural networks) operate on numerical values, and so if our input is numeric, we can pass it through to the model unchanged.



**Scaling.** Often, because the ML framework uses an optimizer that is tuned to work well with numbers in the [–1, 1] range, scaling the numeric values to lie in that range can be beneficial. Four forms of scaling are commonly employed: 

* min-max scaling (normalization)
  $$
  \bar x^{(j)} = \frac{x^{(j)} - min^{(j)}}{max^{(j)} - min^{(j)}}
  $$

* clipping

* z-score normalization (standardization):
  $$
  \hat x^{(j)} = \frac{x^{(j)} - \mu^{(j)}}{\sigma^{(j)}}
  $$

* winsorizing.  

Min-max and clipping tend to work best for uniformly distributed data, and Z-score tends to work best for normally distributed data.

Usually, if your dataset is not too big and you have time, you can try both and see which one performs better for your task. If you don’t have time to run multiple experiments, as a rule of thumb:

* unsupervised learning algorithms, in practice, more often benefit from standardization than from normalization; 
* standardization is also preferred for a feature if the values this feature takes are distributed close to a normal distribution (so-called bell curve); 
* again, standardization is preferred for a feature if it can sometimes have extremely high or low values (outliers); this is because normalization will “squeeze” the normal values into a very small range; 
* in all other cases, normalization is preferable.



**Nonlinear transformations.** What if our data is skewed and neither uniformly distributed nor distributed like a bell curve? In that case, it is better to apply a nonlinear transform to the input before scaling it. One common trick is to take the logarithm of the input value before scaling it. Other common transformations include the sigmoid and polynomial expansions (square, square root, cube, cube root, and so on). 

It can be difficult to devise a linearizing function that makes the distribution look like a bell curve. An easier approach is to bucketize. A principled approach to choosing these buckets is to do *histogram equalization*, where the bins of the histogram are chosen based on quantiles of the raw distribution.

Another method to handle skewed distributions is to use a parametric transformation technique like the *Box-Cox transform*. Box-Cox chooses its single parameter, lambda, to control the “heteroscedasticity” so that the variance no longer depends on the magnitude.



**Array of numbers.** Sometimes, the input data is an array of numbers. If the array is of fixed length, data representation can be rather simple: flatten the array and treat each position as a separate feature.

Common idioms to handle arrays of numbers (variable length or fixed) include the following: 

* Representing the input array in terms of its bulk statistics. For example, we might use the length (that is, count of previous books on topic), average, median, minimum, maximum, and so forth. 
* Representing the input array in terms of its empirical distribution—i.e., by the 10th/20th/... percentile, and so on. 
* If the array is ordered in a specific way (for example, in order of time or by size), representing the input array by the last three or some other fixed number of items. For arrays of length less than three, the feature is padded to a length of three with missing values.



#### Categorical inputs

Because most modern, large-scale machine learning models (random forests, support vector machines, neural networks) operate on numerical values, categorical inputs have to be represented as numbers.

There are several methods to represent categorical variables as numbers:

* Dummy variables - Binary 0-1 variables derived by recording factor data for use in regression and other models.
* Reference coding - The most common type of coding used by statisticians, in which one level of a factor is used as a reference and other factors are compared to that level
* One hot encoder - A common type of coding used in machine learning community in which all factors levels are retained. While useful for a certain machine learning algorithms, this approach is not appropriate for multiple linear regression.
* Deviation coding - A type of coding that compares each level against the overall mean as opposed to the reference level.

**Array of categorical variables.** If the array is of fixed length, we can treat each array position as a separate feature. Common idioms to handle arrays of categorical variables include the following: 

* Counting the number of occurrences of each vocabulary item.  This is now a fixed-length array of numbers that can be flattened and used in positional order. If we have an array where an item can occur only once (for example, of languages a person speaks), or if the feature just indicates presence and not count (such as whether the mother has ever had a Cesarean operation), then the count at each position is 0 or 1, and this is called multi-hot encoding. 
* To avoid large numbers, the relative frequency can be used instead of the count. Empty arrays (first-born babies with no previous siblings) are represented as [0, 0, 0]. In natural language processing, the relative frequency of a word overall is normalized by the relative frequency of documents that contain the word to yield TFIDF (short for term frequency–inverse document frequency). TF-IDF reflects how unique a word is to a document. 
* If the array is ordered in a specific way (e.g., in order of time), representing the input array by the last three items. Arrays shorter than three are padded with missing values. 
* Representing the array by bulk statistics, e.g., the length of the array, the mode (most common entry), the median, the 10th/20th/… percentile, etc.



### Hashed Feature

The Hashed Feature design pattern addresses three possible problems associated with categorical features: incomplete vocabulary, model size due to cardinality, and cold start. It does so by grouping the categorical features and accepting the trade-off of collisions in the data representation.

**Problem.** One-hot encoding a categorical input variable requires knowing the vocabulary beforehand. This is not a problem if the input variable is something like the language a book is written in or the day of the week that traffic level is being predicted.

What if the categorical variable in question is something like the hospital_id of where the baby is born or the physician_id of the person delivering the baby? Categorical variables like these pose a few problems:

* Knowing the vocabulary requires extracting it from the training data. Due to random sampling, it is possible that the training data does not contain all the possible hospitals or physicians. The vocabulary might be incomplete. 
* The categorical variables have high cardinality. Instead of having feature vectors with three languages or seven days, we have feature vectors whose length is in the thousands to millions. Such feature vectors pose several problems in practice. They involve so many weights that the training data may be insufficient. Even if we can train the model, the trained model will require a lot of space to store because the entire vocabulary is needed at serving time. Thus, we may not be able to deploy the model on smaller devices. 
* After the model is placed into production, new hospitals might be built and new physicians hired. The model will be unable to make predictions for these, and so a separate serving infrastructure will be required to handle such cold-start problems.

**Solution.** The Hashed Feature design pattern represents a categorical input variable by doing the following:

1. Converting the categorical input into a unique string.
2. Invoking a deterministic (no random seeds or salt) and portable (so that the same algorithm can be used in both training and serving) hashing algorithm on the string.
3. Taking the remainder when the hash result is divided by the desired number of buckets (A good rule of thumb is to choose the number of hash buckets such that each bucket gets about five entries). Typically, the hashing algorithm returns an integer that can be negative and the modulo of a negative integer is negative. So, the absolute value of the result is taken.

**Why It Works.** 

* Out-of-vocabulary input - Even if an airport with a handful of flights is not part of the training dataset, its hashed feature value will be in some range.
* High cardinality - It’s easy to see that the high cardinality problem is addressed as long as we choose a small enough number of hash buckets. Even if we have millions of airports or hospitals or physicians, we can hash them into a few hundred buckets, thus keeping the system’s memory and model size requirements practical.
* Cold start - The cold-start situation is similar to the out-of-vocabulary situation.

**Trade-Offs and Alternatives.** The key trade-off here is that we lose model accuracy.

* Bucket collision - We are explicitly compromising on the ability to accurately represent the data (with a fixed vocabulary and one-hot encoding) in order to handle out-of-vocabulary inputs, cardinality/model size constraints, and cold-start problems. It is not a free lunch. Do not choose Hashed Feature if you know the vocabulary beforehand, if the vocabulary size is relatively small (in the thousands is acceptable for a dataset with millions of examples), and if cold start is not a concern.
* Skew - The loss of accuracy is particularly acute when the distribution of the categorical input is highly skewed.
* Aggregate feature - In cases where the distribution of a categorical variable is skewed or where the number of buckets is so small that bucket collisions are frequent, we might find it helpful to add an aggregate feature as an input to our model.
* Hyperparameter tuning - Because of the trade-offs with bucket collision frequency, choosing the number of buckets can be difficult. It very often depends on the problem itself. Therefore, we recommend that you treat the number of buckets as a hyperparameter that is tuned.
* Empty hash buckets - Although unlikely, there is a remote possibility that even if we choose 10 hash buckets to represent 347 airports, one of the hash buckets will be empty. Therefore, when using hashed feature columns, it may be beneficial to also use L2 regularization so that the weights associated with an empty bucket will be driven to near-zero. This way, if an out-of-vocabulary airport does fall into an empty bucket, it will not cause the model to become numerically unstable.





### Embeddings



Embeddings are a learnable data representation that map high-cardinality data into a lower-dimensional space in such a way that the information relevant to the learning problem is preserved. Embeddings are at the heart of modern-day machine learning and have various incarnations throughout the field.

**Problem.** For example, what if our dataset consisted of customers’ view history of our video database and our task is to suggest a list of new videos given customers’ previous video interactions? In this scenario, the customer_id field could have millions of unique entries. Similarly, the video_id of previously watched videos could contain thousands of entries as well. One-hot encoding high cardinality categorical features like video_ids or customer_ids as inputs to a machine learning model leads to a sparse matrix that isn’t well suited for a number of machine learning algorithms. The second problem with one-hot encoding is that it treats the categorical variables as being independent. However, the data representation for twins 3 6 should be close to the data representation for triplets and quite far away from the data representation for quintuplets.

**Solution.** The Embeddings design pattern addresses the problem of representing high cardinality data densely in a lower dimension by passing the input data through an embedding layer that has trainable weights. This will map the high dimensional, categorical input variable to a real-valued vector in some lowdimensional space. 

1. Text embeddings - Text provides a natural setting where it is advantageous to use an embedding layer. Given the cardinality of a vocabulary (often on the order of tens of thousands of words), one-hot encoding each word isn’t practical. This would create an incredibly large (high-dimensional) and sparse matrix for training. Also, we’d like similar words to have embeddings close by and unrelated words to be far away in embedding space. Therefore, we use a dense word embedding to vectorize the discrete text input before passing to our model.
2. Image embeddings - For image embeddings, a complex convolutional neural network—like Inception or ResNet—is first trained on a large image dataset, like ImageNet, containing millions of images and thousands of possible classification labels. Then, the last softmax layer is removed from the model. Without the final softmax classifier layer, the model can be used to extract a feature vector for a given input. This feature vector contains all the relevant information of the image so it is essentially a low-dimensional embedding of the input image.

**Why It Works.** The embedding layer is just another hidden layer of the neural network. The weights are then associated to each of the high-cardinality dimensions, and the output is passed through the rest of the network. Therefore, the weights to create the embedding are learned through the process of gradient descent just like any other weights in the neural network. This means that the resulting vector embeddings represent the most efficient low-dimensional representation of those feature values with respect to the learning task.

**Trade-Offs and Alternatives.** The main trade-off with using an embedding is the compromised representation of the data. There is a loss of information involved in going from a high cardinality representation to a lower-dimensional representation. In return, we gain information about closeness and context of the items.

* Choosing the embedding dimension - The lossiness of the representation is controlled by the size of the embedding layer. By choosing a very small output dimension of an embedding layer, too much information is forced into a small vector space and context can be lost. On the other hand, when the embedding dimension is too large, the embedding loses the learned contextual importance of the features.

* Autoencoders - Training embeddings in a supervised way can be hard because it requires a lot of labeled data. Autoencoders provide one way to get around this need for a massive labeled dataset.

* Context language models - Is there an auxiliary learning task that works for text? Context language models like Word2Vec and masked language models like Bidirectional Encoding Representations from Transformers (BERT) change the learning task to a problem so that there is no scarcity of labels.

  Word2Vec is a well-known method for constructing an embedding using shallow neural networks and combining two techniques—Continuous Bag of Words (CBOW) and a skip-gram model—applied to a large corpus of text, such as Wikipedia. While the goal of both models is to learn the context of a word by mapping input word(s) to the target word(s) with an intermediate embedding layer, an auxiliary goal is achieved that learns low-dimensional embeddings that best capture the context of words. The resulting word embeddings learned through Word2Vec capture the semantic relationships between words so that, in the embedding space, the vector representations maintain meaningful distance and directionality.

  BERT is trained using a masked language model and next sentence prediction. For a masked language model, words are randomly masked from text and the model guesses what the missing word(s) are. Next sentence prediction is a classification task where the model predicts whether or not two sentences followed each other in the original text. So any corpus of text is suitable as a labeled dataset. BERT was initially trained on all of the English Wikipedia and BooksCorpus. Despite learning on these auxiliary tasks, the learned embeddings from BERT or Word2Vec have proven very powerful when used on other downstream training tasks. The word embeddings learned by Word2Vec are the same regardless of the sentence where the word appears. However, the BERT word embeddings are contextual, meaning the embedding vector is different depending on the context of how the word is used.

  A pre-trained text embedding, like Word2Vec, NNLM, GLoVE, or BERT, can be added to a machine learning model to process text features in conjunction with structured inputs and other learned embeddings from our customer and video dataset.

* Embeddings in a data warehouse - Machine learning on structured data is best carried out directly in SQL on a data warehouse. This avoids the need to export data out of the warehouse and mitigates problems with data privacy and security. Many problems, however, require a mix of structured data and natural language text or image data. In data warehouses, natural language text (such as reviews) is stored directly as columns, and images are typically stored as URLs to files in a cloud storage bucket. In these cases, it simplifies later machine learning to additionally store the embeddings of the text columns or of the images as array type columns. Doing so will enable the easy incorporation of such unstructured data into machine learning models.





### Feature Cross



The Feature Cross design pattern helps models learn relationships between inputs faster by explicitly making each combination of input values a separate feature.

A feature cross is a synthetic feature formed by concatenating two or more categorical features in order to capture the interaction between them. By joining two features in this way, it is possible to encode nonlinearity into the model, which can allow for predictive abilities beyond what each of the features would have been able to provide individually. Consequently, feature crosses can speed up model training (less expensive) and reduce model complexity (less training data is needed).

**Why It Works.** Feature crosses provide a valuable means of feature engineering. They provide more complexity, more expressivity, and more capacity to simple models.

**Trade-Offs and Alternatives.** While we discussed feature crosses as a way of handling categorical variables, they can be applied, with a bit of preprocessing, to numerical features also. Feature crosses cause sparsity in models and are often used along with techniques that counteract that sparsity.

* Handling numerical features - We would never want to create a feature cross with a continuous input. Remember, if one input takes m possible values and another input takes n possible values, then the feature cross of the two would result in m*n elements. Instead, if our data is continuous, then we can bucketize the data to make it categorical before applying a feature cross.
* Handling high cardinality - Because the cardinality of resulting categories from a feature cross increases multiplicatively with respect to the cardinality of the input features, feature crosses lead to sparsity in our model inputs. It can be useful to pass a feature cross through an Embedding layer to create a lower-dimensional representation. Because the Embeddings design pattern allows us to capture closeness relationships, passing the feature cross through an embedding layer allows the model to generalize how certain feature crosses coming from pairs of hour and day combinations affect the output of the model.
* Need for regularization - When crossing two categorical features both with large cardinality, we produce a cross feature with multiplicative cardinality. Naturally, given more categories for an individual feature, the number of categories in a feature cross can increase dramatically. If this gets to the point where individual buckets have too few items, it will hinder the model’s ability to generalize. For this reason, it is advisable to pair feature crosses with L1 regularization, which encourages sparsity of features, or L2 regularization, which limits overfitting. This allows our model to ignore the extraneous noise generated by the many synthetic features and combat overfitting.





### Multimodal Input



The Multimodal Input design pattern addresses the problem of representing different types of data or data that can be expressed in complex ways by concatenating all the available data representations.

**Problem.** Typically, an input to a model can be represented as a number or as a category, an image, or free-form text. Many off-the-shelf models are defined for specific types of input only—a standard image classification model such as Resnet-50, for example, does not have the ability to handle inputs other than images.

This problem also occurs when training a structured data model where one of the inputs is free-form text. Unlike numerical data, images and text cannot be fed directly into a model. As a result, we’ll need to represent image and text inputs in a way our model can understand (usually using the Embeddings design pattern), then combine these inputs with other tabular features. 

**Solution. ** Let’s take the example with text from a restaurant review combined with tabular metadata about the meal referenced by the review. We’ll first combine the numerical and categorical features. There are three possible options for meal_type, so we can turn this into a one-hot encoding and will represent dinner as [0, 0, 1]. With this categorical feature represented as an array, we can now combine it with meal_total by adding the price of the meal as the fourth element of the array: [0, 0, 1, 30.5]. The Embeddings design pattern is a common approach to encoding text for machine learning models. If our model had only text, we could represent it as an embedding layer. We now need to concatenate these three numbers, which form the sentence embedding of the review with the earlier inputs.

```python
embedding_input = Input(shape=(30,))
embedding_layer = Embedding(batch_size, 64)(embedding_input)
embedding_layer = Flatten()(embedding_layer)
embedding_layer = Dense(3, activation='relu')(embedding_layer)

tabular_input = Input(shape=(4,))
tabular_layer = Dense(32, activation='relu')(tabular_input)

merged_input = keras.layers.concatenate([embedding_layer, tabular_layer])
merged_dense = Dense(16)(merged_input)
output = Dense(1)(merged_dense)
model = Model(inputs=[embedding_input, tabular_input], outputs=output)
```



### Data imputation

1. One data imputation technique consists in replacing the missing value of a feature by an **average** value of this feature in the dataset. 

   **Плюсы:**

   - Просто и быстро.
   - Хорошо работает на небольших наборах численных данных.

   **Минусы:**

   - Значения вычисляются независимо для каждого столбца, так что корреляции между параметрами не учитываются.
   - Не работает с категориальными переменными.
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

     

6. Finally, if you have a significantly large dataset and just a few features with missing values, you can increase the dimensionality of your feature vectors by adding a **binary indicator** feature for each feature with missing values. Let’s say feature j = 12 in your D-dimensional dataset has missing values. For each feature vector x, you then add the feature j = D + 1 which is equal to 1 if the value of feature 12 is present in x and 0 otherwise. The missing feature value then can be replaced by 0 or any number of your choice.



### Feature Selection

A classic sequential feature selection algorithm is sequential backward selection (SBS), which aims to reduce the dimensionality of the initial feature subspace with a minimum decay in the performance of the classifier to improve upon computational efficiency.

The idea behind the SBS algorithm is quite simple: SBS sequentially removes features from the full feature subset until the new feature subspace contains the desired number of features. In order to determine which feature is to be removed at each stage, we need to define the criterion function, J, that we want to minimize. 

The criterion calculated by the criterion function can simply be the difference in performance of the classifier before and after the removal of a particular feature. Then, the feature to be removed at each stage can simply be defined as the feature that maximizes this criterion; or in more simple terms, at each stage we eliminate the feature that causes the least performance loss after removal. Based on the preceding definition of SBS, we can outline the algorithm in four simple steps: 

1. Initialize the algorithm with k = d, where d is the dimensionality of the full feature space, 𝑿𝑑. 
2. Determine the feature, 𝒙−, that maximizes the criterion: 𝒙− = argmax 𝐽(𝑿𝑘 − 𝒙), where 𝒙 ∈ 𝑿𝑘. 
3. Remove the feature, 𝒙−, from the feature set: 𝑿𝑘−1 = 𝑿𝑘 − 𝒙− ; 𝑘 = 𝑘 − 1. 
4. Terminate if k equals the number of desired features; otherwise, go to step 2

Also we can select features use random forest importance, l1 regularization.





### Feature importance

There are two ways to measure variable importance: 

1. By the decrease in accuracy of the model if the values of a variable are randomly permuted (type=1). Randomly permuting the values has the effect of removing all predictive power for that variable. The accuracy is computed from the out-of-bag data (so this measure is effectively a cross-validated estimate). 
2. By the mean decrease in the Gini impurity score for all of the nodes that were split on a variable (type=2). This measures how much improvement to the purity of the nodes that variable contributes. This measure is based on the training set, and therefore less reliable than a measure calculated on out-of-bag data.

Ensemble models improve model accuracy by combining the results from many models. Bagging is a particular type of ensemble model based on fitting many models to bootstrapped samples of the data and averaging the models. Random forest is a special type of bagging applied to decision trees. In addition to resampling the data, the random forest algorithm samples the predictor variables when splitting the trees. A useful output from the random forest is a measure of variable importance that ranks the predictors in terms of their contribution to model accuracy. The random forest has a set of hyperparameters that should be tuned using cross-validation to avoid overfitting.









## Problem Representation Design Patterns



The input and the output types are two key factors impacting the model architecture. For instance, the output in supervised machine learning problems can vary depending on whether the problem being solved is a classification or regression problem. Special neural network layers exist for specific types of input data: convolutional layers for images, speech, text, and other data with spatiotemporal correlation, recurrent networks for sequential data, and so on. A huge literature has arisen around special techniques such as max pooling, attention, and so forth on these types of layers. In addition, special classes of solutions have been crafted for commonly occurring problems like recommendations (such as matrix factorization) or time-series forecasting (for example, ARIMA). Finally, a group of simpler models together with common idioms can be used to solve more complex problems—for example, text generation often involves a classification model whose outputs are postprocessed using a beam search algorithm.



### Reframing



The Reframing design pattern refers to changing the representation of the output of a machine learning problem. For example, we could take something that is intuitively a regression problem and instead pose it as a classification problem (and vice versa).

**Problem.** The first step of building any machine learning solution is framing the problem. Is this a supervised learning problem? Or unsupervised? The answers to these questions must be considered in context with the training data, the task at hand, and the metrics for success.

For example, suppose we wanted to build a machine learning model to predict future rainfall amounts in a given location. Starting broadly, would this be a regression or classification task? Well, since we’re trying to predict rainfall amount (for example, 0.3 cm), it makes sense to consider this as a time-series forecasting problem: given the current and historical climate and weather patterns, what amount of rainfall should we expect in a given area in the next 15 minutes? Alternately, because the label (the amount of rainfall) is a real number, we could build a regression model. As we start to develop and train our model, we find (perhaps not surprisingly) that weather prediction is harder than it sounds. Our predicted rainfall amounts are all off because, for the same set of features, it sometimes rains 0.3 cm and other times it rains 0.5 cm. What should we do to improve our predictions? Should we add more layers to our network? Or engineer more features? Perhaps more data will help? Maybe we need a different loss function? Any of these adjustments could improve our model. But wait. Is regression the only way we can pose this task? Perhaps we can reframe our machine learning objective in a way that improves our task performance.

**Solution.** The core issue here is that rainfall is probabilistic. For the same set of features, it sometimes rains 0.3 cm and other times it rains 0.5 cm. Yet, even if a regression model were able to learn the two possible amounts, it is limited to predicting only a single number. Instead of trying to predict the amount of rainfall as a regression task, we can reframe our objective as a classification problem. There are different ways this can be accomplished. One approach is to model a discrete probability distribution. Instead of predicting rainfall as a real-valued output, we model the output as a multiclass classification giving the probability that the rainfall in the next 15 minutes is within a certain range of rainfall amounts.

Both the regression approach and this reframed-as-classification approach give a prediction of the rainfall for the next 15 minutes. However, the classification approach allows the model to capture the probability distribution of rainfall of different quantities instead of having to choose the mean of the distribution.

**Why It Works.** Instead of learning a single real number, we relax our prediction target to be instead a discrete probability distribution. We lose a little precision due to bucketing, but gain the expressiveness of a full probability density function (PDF). The discretized predictions provided by the classification model are more adept at learning a complex target than the more rigid regression model.

**Trade-Offs and Alternatives.** There is rarely just one way to frame a problem, and it is helpful to be aware of any trade-offs or alternatives of a given implementation. 

* Bucketized outputs - The typical approach to reframing a regression task as a classification is to bucketize the output values.

* Other ways of capturing uncertainty - A simple approach is to carry out quantile regression. For example, instead of predicting just the mean, we can estimate the conditional 10th, 20th, 30th, …, 90th percentile of what needs to be predicted.

* Precision of predictions - When thinking of reframing a regression model as a multiclass classification, the width of the bins for the output label governs the precision of the classification model.

* Restricting the prediction range - Another reason to reframe the problem is when it is essential to restrict the range of the prediction output. Let’s say, for example, that realistic output values for a regression problem are in the range [3, 20]. If we train a regression model where the output layer is a linear activation function, there is always the possibility that the model predictions will fall outside this range. One way to limit the range of the output is to reframe the problem.

* Label bias - It is important to consider the nature of the target label when reframing the problem. For example, suppose we reframed our recommendation model to a classification task that predicts the likelihood a user will click on a certain video thumbnail. This seems like a reasonable reframing since our goal is to provide content a user will select and watch. But be careful. This change of label is not actually in line with our prediction task. By optimizing for user clicks, our model will inadvertently promote click bait and not actually recommend content of use to the user. Instead, a more advantageous label would be video watch time, reframing our recommendation as a regression instead.

* Multitask learning - One alternative to reframing is multitask learning. Instead of trying to choose between regression or classification, do both! Generally speaking, multitask learning refers to any machine learning model in which more than one loss function is optimized. This can be accomplished in many different ways, but the two most common forms of multi task learning in neural networks is through hard parameter sharing and soft parameter sharing.

  Parameter sharing refers to the parameters of the neural network being shared between the different output tasks, such as regression and classification. Hard parameter sharing occurs when the hidden layers of the model are shared between all the output tasks. In soft parameter sharing, each label has its own neural network with its own parameters, and the parameters of the different models are encouraged to be similar through some form of regularization.



### Multilabel



The Multilabel design pattern refers to problems where we can assign more than one label to a given training example. For neural networks, this design requires changing the activation function used in the final output layer of the model and choosing how our application will parse model output. Note that this is different from multiclass classification problems, where a single example is assigned exactly one label from a group of many (> 1) possible classes. You may also hear the Multilabel design pattern referred to as multilabel, multiclass classification since it involves choosing more than one label from a group of more than one possible class. 

**Problem.** The Multilabel design pattern exists for models trained on all data modalities. For image classification, in the cat, dog, rabbit example, we could instead use training images that each depicted multiple animals, and could therefore have multiple labels. For text models, we can imagine a few scenarios where text can be labeled with multiple tags. This design pattern can also apply to tabular datasets. Imagine a healthcare dataset with various physical characteristics for each patient, like height, weight, age, blood pressure, and more. This data could be used to predict the presence of multiple conditions. For example, a patient could show risk of both heart disease and diabetes.

**Solution.** The solution for building models that can assign more than one label to a given training example is to use the sigmoid activation function in our final output layer. Rather than generating an array where all values sum to 1 (as in softmax), each individual value in a sigmoid array is a float between 0 and 1. That is to say, when implementing the Multilabel design pattern, our label needs to be multi-hot encoded. The length of the multi-hot array corresponds with the number of classes in our model, and each output in this label array will be a sigmoid value.

**Trade-Offs and Alternatives.** There are several special cases to consider when following the Multilabel design pattern and using sigmoid output. Next, we’ll explore how to structure models that have two possible label classes, how to make sense of sigmoid results, and other important considerations for Multilabel models.

* Sigmoid output for models with two classes.
* Parsing sigmoid results - To extract the predicted label for a model with softmax output, we can simply take the argmax (highest value index) of the output array to get the predicted class. Parsing sigmoid outputs is less straightforward. Instead of taking the class with the highest predicted probability, we need to evaluate the probability of each class in our output layer and consider the probability threshold for our use case.
* Dataset considerations - When dealing with single-label classification tasks, we can ensure our dataset is balanced by aiming for a relatively equal number of training examples for each class. Building a balanced dataset is more nuanced for the Multilabel design pattern.
* Inputs with overlapping labels - The Multilabel design pattern is also useful in cases where input data occasionally has overlapping labels. Let’s take an image model that’s classifying clothing items for a catalog as an example. If we have multiple people labeling each image in the training dataset, one labeler may label an image of a skirt as “maxi skirt,” while another identifies it as “pleated skirt.” Both are correct. However, if we build a multiclass classification model on this data, passing it multiple examples of the same image with different labels, we’ll likely encounter situations where the model labels similar images differently when making predictions.





### Cascade



The Cascade design pattern addresses situations where a machine learning problem can be profitably broken into a series of ML problems. Such a cascade often requires careful design of the ML experiment.

**Problem.** What happens if we need to predict a value during both usual and unusual activity? The model will learn to ignore the unusual activity because it is rare If the unusual activity is also associated with abnormal values, then trainability suffers.

For example, suppose we are trying to train a model to predict the likelihood that a customer will return an item that they have purchased. If we train a single model, the resellers’ return behavior will be lost because there are millions of retail buyers (and retail transactions) and only a few thousand resellers. 

One way to solve this problem is to overweight the reseller instances when training the model. This is suboptimal because we need to get the more common retail buyer use case as correct as possible. An intuitive way to address this problem is by using the Cascade design pattern. We break the problem into four parts:

1. Predicting whether a specific transaction is by a reseller 
2. Training one model on sales to retail buyers 
3. Training the second model on sales to resellers 
4. In production, combining the output of the three separate models to predict return likelihood for every item purchased and the probability that the transaction is by a reseller

This allows for the possibility of different decisions on items likely to be returned depending on the type of buyer and ensures that the models in steps 2 and 3 are as accurate as possible on their segment of the training data.

The problem comes during prediction. At prediction time, we don’t have true labels, just the output of the first classification model. Based on the output of the first model, we will have to determine which of the two sales models we invoke. The problem is that we are training on labels, but at inference time, we will have to make decisions based on predictions. And predictions have errors. So, the second and third models will be required to make predictions on data that they might have never seen during training.

**Solution. ** Any machine learning problem where the output of the one model is an input to the following model or determines the selection of subsequent models is called a cascade. Special care has to be taken when training a cascade of ML models.

In practice, it can become hard to keep a Cascade workflow straight. Rather than train the models individually, it is better to automate the entire workflow using the Workflow Pipelines pattern. The key is to ensure that training datasets for the two downstream models are created each time the experiment is run based on the predictions of upstream models.

Although we introduced the Cascade pattern as a way of predicting a value during both usual and unusual activity, the Cascade pattern’s solution is capable of addressing a more general situation. 

**Trade-Offs and Alternatives. **Cascade is not necessarily a best practice. It adds quite a bit of complexity to your machine learning workflows and may actually result in poorer performance.

* Deterministic inputs - Splitting an ML problem is usually a bad idea, since an ML model can/should learn combinations of multiple factors.
* Single model - The Cascade design pattern should not be used for common scenarios where a single model will suffice. For example, suppose we are trying to learn a customer’s propensity to buy. We may think we need to learn different models for people who have been comparison shopping versus those who aren’t. We don’t really know who has been comparison shopping, but we can make an educated guess based on the number of visits, how long the item has been in the cart, and so on. This problem does not need the Cascade design pattern because it is common enough (a large fraction of customers will be comparison shopping) that the machine learning model should be able to learn it implicitly in the course of training. 
* Internal consistency - The Cascade is needed when we need to maintain internal consistency amongst the predictions of multiple models. Note that we are trying to do more than just predict the unusual activity. We are trying to predict returns, considering that there will be some reseller activity also. If the task is only to predict whether or not a sale is by a reseller, we’d use the Rebalancing pattern. The reason to use Cascade is that the imbalanced label output is needed as an input to subsequent models and is useful in and of itself.
* Pre-trained models - The Cascade is also needed when we wish to reuse the output of a pre-trained model as an input into our model.





###  Neutral Class



In many classification situations, creating a neutral class can be helpful. For example, instead of training a binary classifier that outputs the probability of an event, train a three-class classifier that outputs disjoint probabilities for Yes, No, and Maybe. Disjoint here means that the classes do not overlap. A training pattern can belong to only one class, and so there is no overlap between Yes and Maybe, for example. The Maybe in this case is the neutral class.

**Problem.** Imagine that we are trying to create a model that provides guidance on pain relievers. There are two choices, ibuprofen and acetaminophen, and it turns out in our historical dataset that acetaminophen tends to be prescribed preferentially to patients at risk of stomach problems, and ibuprofen tends to be prescribed preferentially to patients at risk of liver damage. Beyond that, things tend to be quite random; some physicians default to acetaminophen and others to ibuprofen.

Training a binary classifier on such a dataset will lead to poor accuracy because the model will need to get the essentially arbitrary cases correct.

**Solution.** Imagine a different scenario. Suppose the electronic record that captures the doctor’s prescriptions also asks them whether the alternate pain medication would be acceptable. If the doctor prescribes acetaminophen, the application asks the doctor whether the patient can use ibuprofen if they already have it in their medicine cabinet. Based on the answer to the second question, we have a neutral class. The prescription might still be written as “acetaminophen,” but the record captures that the doctor was neutral for this patient. Note that this fundamentally requires us to design the data collection appropriately—we cannot manufacture a neutral class after the fact. We have to correctly design the machine learning problem. Correct design, in this case, starts with how we pose the problem in the first place.

**Trade-Offs and Alternatives.** The Neutral Class design pattern is one to keep in mind at the beginning of a machine learning problem. Collect the right data, and we can avoid a lot of sticky problems down the line. Here are a few situations where having a neutral class can be helpful.

* When human experts disagree - The neutral class is helpful in dealing with disagreements among human experts.
* Customer satisfaction - The need for a neutral class also arises with models that attempt to predict customer satisfaction. If the training data consists of survey responses where customers grade their experience on a scale of 1 to 10, it might be helpful to bucket the ratings into three categories: 1 to 4 as bad, 8 to 10 as good, and 5 to 7 is neutral. If, instead, we attempt to train a binary classifier by thresholding at 6, the model will spend too much effort trying to get essentially neutral responses correct.



### Rebalancing



The Rebalancing design pattern provides various approaches for handling datasets that are inherently imbalanced. By this we mean datasets where one label makes up the majority of the dataset, leaving far fewer examples of other labels.

**Problem.** Machine learning models learn best when they are given a similar number of examples for each label class in a dataset. Many real-world problems, however, are not so neatly balanced. Take for example a fraud detection use case, where you are building a model to identify fraudulent credit card transactions.

**Solution. ** First, since accuracy can be misleading on imbalanced datasets, it’s important to choose an appropriate evaluation metric when building our model. Then, there are various techniques we can employ for handling inherently imbalanced datasets at both the dataset and model level. 

* Downsampling changes the balance of our underlying dataset, while weighting changes how our model handles certain classes. Downsampling is usually combined with the Ensemble pattern, following these steps: 

  1. Downsample the majority class and use all the instances of the minority class. 
  2. Train a model and add it to the ensemble. 
  3. Repeat. 

  During inference, take the median output of the ensemble models.

* Weighted classes - By weighting classes, we tell our model to treat specific label classes with more importance during training. We’ll want our model to assign more weight to examples from the minority class. In Keras, we can pass a class_weights parameter to our model when we train it with fit().

* Upsampling duplicates examples from our minority class, and often involves applying augmentations to generate additional samples. This is often done in combination with downsampling the majority class. This approach—combining downsampling and upsampling— Synthetic Minority Over-sampling Technique.  SMOTE provides an algorithm that constructs these synthetic examples by analyzing the feature space of minority class examples in the dataset and then generates similar examples within this feature space using a nearest neighbors approach. Depending on how many similar data points we choose to consider at once (also referred to as the number of nearest neighbors), the SMOTE approach randomly generates a new minority class example between these points.



















## Model Training Patterns



### Useful Overfitting



Useful Overfitting is a design pattern where we forgo the use of generalization mechanisms because we want to intentionally overfit on the training dataset. In situations where overfitting can be beneficial, this design pattern recommends that we carry out machine learning without regularization, dropout, or a validation dataset for early stopping.

**Problem.** Consider a situation of simulating the behavior of physical or dynamical systems like those found in climate science, computational biology, or computational finance. In such systems, the time dependence of observations can be described by a mathematical function or set of partial differential equations (PDEs). Although the equations that govern many of these systems can be formally expressed, they don’t have a closed-form solution. Instead, classical numerical methods have been developed to approximate solutions to these systems. Unfortunately, for many real-world applications, these methods can be too slow to be used in practice.

**Solution.** In this scenario, there is no “unseen” data that needs to be generalized to, since all possible inputs have been tabulated. When building a machine learning model to learn such a physics model or dynamical system, there is no such thing as overfitting. The basic machine learning training paradigm is slightly different. Here, there is some physical phenomenon that you are trying to learn that is governed by an underlying PDE or system of PDEs. Machine learning merely provides a data-driven approach to approximate the precise solution, and concepts like overfitting must be reevaluated. We want our ML model to fit the training data as perfectly as possible, to “overfit.”

**Why It Works.** One bit of intuition as to why this works comes from the Uniform Approximation Theorem of deep learning, which, loosely put, states that any function (and its derivatives) can be approximated by a neural network with at least one hidden layer and any “squashing” activation function, like sigmoid. 

Overfitting is useful when the following two conditions are met:

1. There is no noise, so the labels are accurate for all instances. 
2. You have the complete dataset at your disposal (you have all the examples there are). In this case, overfitting becomes interpolating the dataset.

**Trade-Offs and Alternatives.** If the full input space can be tabulated, overfitting is not a concern because there is no unseen data. However, the Useful Overfitting design pattern is useful beyond this narrow use case. In many real-world situations, even if one or more of these conditions has to be relaxed, the concept that overfitting can be useful remains valid.

* Interpolation and chaos theory - The machine learning model essentially functions as an approximation to a lookup table of inputs to outputs. If the lookup table is small, just use it as a lookup table! There is no need to approximate it by a machine learning model.

  Machine learning models interpolate by weighting unseen values by the distance of these unseen values from training examples. Such interpolation works only if the underlying system is not chaotic. In chaotic systems, even if the system is deterministic, small differences in initial conditions can lead to dramatically different outcomes.

* Monte Carlo methods - In reality, tabulating all possible inputs might not be possible, and you might take a Monte Carlo approach of sampling the input space to create the set of inputs, especially where not all possible combinations of inputs are physically possible. In such cases, overfitting is technically possible. However, even here, you can see that the ML model will be interpolating between known answers. The calculation is always deterministic, and it is only the input points that are subject to random selection. Therefore, these known answers do not contain noise, and because there are no unobserved variables, errors at unsampled points will be strictly bounded by the model complexity.

* Data-driven discretizations - Although deriving a closed-form solution is possible for some PDEs, determining solutions using numerical methods is more common. One common approach is to use finite difference methods, similar to Runge-Kutta methods, for solving ordinary differential equations. This is typically done by discretizing the differential operator of the PDE and finding a solution to the discrete problem on a spatio-temporal grid of the original domain. However, when the dimension of the problem becomes large, this meshbased approach fails dramatically due to the curse of dimensionality.

  It is possible to use machine learning (rather than Monte Carlo methods) to select the sampling points to create data-driven discretizations of PDEs.

* Unbounded domains - The Monte Carlo and data-driven discretization methods both assume that sampling the entire input space, even if imperfectly, is possible. That’s why the ML model was treated as an interpolation between known points. Generalization and the concern of overfitting become difficult to ignore whenever we are unable to sample points in the full domain of the function —for example, for functions with unbounded domains or projections along a time axis into the future. In these settings, it is important to consider overfitting, underfitting, and generalization error.

* Distilling knowledge of neural network - Another situation where overfitting is warranted is in distilling, or transferring knowledge, from a large machine learning model into a smaller one. Knowledge distillation is useful when the learning capacity of the large model is not fully utilized.

  While the smaller model has enough capacity to represent the knowledge, it may not have enough capacity to learn the knowledge efficiently. The solution is to train the smaller model on a large amount of generated data that is labeled by the larger model. The smaller model learns the soft output of the larger model, instead of actual labels on real data.





### Checkpoints



**Problem.** When we have training that takes long time, the chances of machine failure are uncomfortably high. If there is a problem, we’d like to be able to resume from an intermediate point, instead of from the very beginning.

**Solution.** At the end of every epoch, we can save the model state. Then, if the training loop is interrupted for any reason, we can go back to the saved model state and restart. However, when doing this, we have to make sure to save the intermediate model state, not just the model.

An exported model does not contain which epoch and batch number the model is currently processing, which is obviously important in order to resume training. But there is more information that a model training loop can contain. In order to carry out gradient descent effectively, the optimizer might be changing the learning rate on a schedule. This learning rate state is not present in an exported model. Additionally, there might be stochastic behavior in the model, such as dropout. This is not captured in the exported model state either. Models like recurrent neural networks incorporate history of previous input values. In general, the full model state can be many times the size of the exported model.

**Trade-Offs and Alternatives.** 

* Early stopping - In general, the longer you train, the lower the loss on the training dataset. However, at some point, the error on the validation dataset might stop decreasing. If you are starting to overfit to the training dataset, the validation error might even start to increase. In such cases, it can be helpful to look at the validation error at the end of every epoch and stop the training process when the validation error is more than that of the previous epoch.

* Fine-tuning - In a well-behaved training loop, gradient descent behaves such that you get to the neighborhood of the optimal error quickly on the basis of the majority of your data, then slowly converge toward the lowest error by optimizing on the corner cases. Now, imagine that you need to periodically retrain the model on fresh data. You typically want to emphasize the fresh data, not the corner cases from last month. You are often better off resuming your training, not from the last checkpoint, but the "elbow loss" checkpoint.

* Redefining an epoch - using epochs on large datasets remains a bad idea. To see why, imagine that you have a training dataset with one million examples. It can be tempting to simply go through this dataset 15 times (for example) by setting the number of epochs to 15. There are several problems with this:

  * The number of epochs is an integer, but the difference in training time between processing the dataset 14.3 times and 15 times can be hours. If the model has converged after having seen 14.3 million examples, you might want to exit and not waste the computational resources necessary to process 0.7 million more examples. 
  * You checkpoint once per epoch, and waiting one million examples between checkpoints might be way too long. For resilience, you might want to checkpoint more often. 
  * Datasets grow over time. If you get 100,000 more examples and you train the model and get a higher error, is it because you need to do an early stop, or is the new data corrupt in some way? You can’t tell because the prior training was on 15 million examples and the new one is on 16.5 million examples. 
  * In distributed, parameter-server training with data parallelism and proper shuffling, the concept of an epoch is not clear anymore. Because of potentially straggling workers, you can only instruct the system to train on some number of mini-batches.

  Instead of training for 15 epochs, we might decide to train for 143,000 steps where the batch_size is 100. What happens when we get 100,000 more examples? Easy! We add it to our data warehouse but do not update the code. Our code will still want to process 143,000 steps, and it will get to process that much data, except that 10% of the examples it sees are newer. If the model converges, great. If it doesn’t, we know that these new data points are the issue because we are not training longer than we were before. By keeping the number of steps constant, we have been able to separate out the effects of new data from training on more data. Once we have trained for 143,000 steps, we restart the training and run it a bit longer (say, 10,000 steps), and as long as the model continues to converge, we keep training it longer. Then, we update the number 143,000 in the code above (in reality, it will be a parameter to the code) to reflect the new number of steps. This all works fine, until you want to do hyperparameter tuning. When you do hyperparameter tuning, you will want to want to change the batch size. Unfortunately, if you change the batch size to 50, you will find yourself training for half the time because we are training for 143,000 steps, and each step is only half as long as before. 





### Transfer Learning



In Transfer Learning, we take part of a previously trained model, freeze the weights, and incorporate these nontrainable layers into a new model that solves a similar problem, but on a smaller dataset.

**Problem.** Training custom ML models on unstructured data requires extremely large datasets, which are not always readily available. 

**Solution.** With the Transfer Learning design pattern, we can take a model that has been trained on the same type of data for a similar task and apply it to a 2 specialized task using our own custom data.

Transfer learning in neural networks works like this. 

1. You build a deep model on the original big dataset (wild animals). 
2. You compile a much smaller labeled dataset for your second model (domestic animals). 
3. You remove the last one or several layers from the first model. Usually, these are layers responsible for the classification or regression; they usually follow the embedding layer. 
4. You replace the removed layers with new layers adapted for your new problem. 
5. You “freeze” the parameters of the layers remaining from the first model. 
6. You use your smaller labeled dataset and gradient descent to train the parameters of only the new layers.

**Trade-Offs and Alternatives.** 

* Fine-tuning versus feature extraction - Feature extraction describes an approach to transfer learning where you freeze the weights of all layers before the bottleneck layer and train the following layers on your own data and labels. Another option is instead fine-tuning the weights of the pre-trained model’s layers. With fine-tuning, you can either update the weights of each layer in the pre-trained model, or just a few of the layers right before the bottleneck. 
* Focus on image and text models - transfer learning is primarily for cases where you can apply a similar task to the same data domain. Models trained with tabular data, however, cover a potentially infinite number of possible prediction tasks and data types.



### Distribution Strategy



In Distribution Strategy, the training loop is carried out at scale over multiple workers, often with caching, hardware acceleration, and parallelization.

**Problem.** These days, it’s common for large neural networks to have millions of parameters and be trained on massive amounts of data. In fact, it’s been shown that increasing the scale of deep learning, with respect to the number of training examples, the number of model parameters, or both, drastically improves model performance. However, as the size of models and data increases, the computation and memory demands increase proportionally, making the time it takes to train these models one of the biggest problems of deep learning.

**Solution.** One way to accelerate training is through distribution strategies in the training loop. There are different distribution techniques, but the common idea is to split the effort of training the model across multiple machines. There are two ways this can be done: data parallelism and model parallelism. In data parallelism, computation is split across different machines and different workers train on different subsets of the training data. In model parallelism, the model is split and different workers carry out the computation for different parts of the model.

To implement data parallelism, there must be a method in place for different workers to compute gradients and share that information to make updates to the model parameters. This ensures that all workers are consistent and each gradient step works to train the model. Broadly speaking, data parallelism can be carried out either synchronously or asynchronously.

* Synchronous training - In synchronous training, the workers train on different slices of input data in parallel and the gradient values are aggregated at the end of each training step. This is performed via an all-reduce algorithm. This means that each worker, typically a GPU, has a copy of the model on device and, for a single stochastic gradient descent (SGD) step, a mini-batch of data is split among each of the separate workers. Each device performs a forward pass with their portion of the mini-batch and computes gradients for each parameter of the model. These locally computed gradients are then collected from each device and aggregated (for example, averaged) to produce a single gradient update for each parameter. A central server holds the most current copy of the model parameters and performs the gradient step according to the gradients received from the multiple workers. Once the model parameters are updated according to this aggregated gradient step, the new model is sent back to the workers along with another split of the next mini-batch, and the process repeats.

* Asynchronous training - In asynchronous training, the workers train on different slices of the input data independently, and the model weights and parameters are updated asynchronously, typically through a parameter server architecture. This means that no one worker waits for updates to the model from any of the other workers. In the parameter-server architecture, there is a single parameter server that manages the current values of the model weights.

  As with synchronous training, a mini-batch of data is split among each of the separate workers for each SGD step. Each device performs a forward pass with their portion of the mini-batch and computes gradients for each parameter of the model. Those gradients are sent to the parameter server, which performs the parameter update and then sends the new model parameters back to the worker with another split of the next mini-batch. The key difference between synchronous and asynchronous training is that the parameter server does not do an all-reduce. Instead, it computes the new model parameters periodically based on whichever gradient updates it received since the last computation. Typically, asynchronous distribution achieves higher throughput than synchronous training because a slow worker doesn’t block the progression of training steps.

**Trade-Offs and Alternatives.** 

* Model parallelism - In some cases, the neural network is so large it cannot fit in the memory of a single device; By partitioning parts of a network and their associated computations across multiple cores, the computation and memory workload is distributed across multiple devices. Each device operates over the same mini-batch of data during training, but carries out computations related only to their separate components of the model.
* ASICs for better performance at lower cost - Another way to speed up the training process is by accelerating the underlying hardware, such as by using application-specific integrated circuits (ASICs). In machine learning, this refers to hardware components designed specifically to optimize performance on the types of large matrix computations at the heart of the training loop. TPUs in Google Cloud are ASICs that can be used for both model training and making predictions.
* Choosing a batch size - Particular to synchronous data parallelism, when the model is particularly large, it’s better to decrease the total number of training iterations because each training step requires the updated model to be shared among different workers, causing a slowdown for transfer time. Thus, it’s important to increase the mini-batch size as much as possible so that the same performance can be met with fewer steps.





## Design Patterns for Resilient Serving



The purpose of a machine learning model is to use it to make inferences on data it hasn’t seen during training. Therefore, once a model has been trained, it is typically deployed into a production environment and used to make predictions in response to incoming requests. Software that is deployed into production environments is expected to be resilient and require little in the way of human intervention to keep it running.

The Stateless Serving Function design pattern allows the serving infrastructure to scale and handle thousands or even millions of prediction requests per second. The Batch Serving design pattern allows the serving infrastructure to asynchronously handle occasional or periodic requests for millions to billions of predictions. These patterns are useful beyond resilience in that they reduce coupling between creators and users of machine learning models. The Continued Model Evaluation design pattern handles the common problem of detecting when a deployed model is no longer fit-for-purpose. The Two-Phase Predictions design pattern provides a way to address the problem of keeping models sophisticated and performant when they have to be deployed onto distributed devices. The Keyed Predictions design pattern is a necessity to scalably implement several of the design patterns discussed in this chapter.



### Stateless Serving Function



The Stateless Serving Function design pattern makes it possible for a production ML system to synchronously handle thousands to millions of prediction requests per second. The production ML system is designed around a stateless function (A stateless function is a function whose outputs are determined purely by its inputs.) that captures the architecture and weights of a trained model.

**Problem.** Let’s take a text classification model that uses, as its training data, movie reviews from the Internet Movie Database (IMDb). For the initial layer of the model, we will use a pre-trained embedding that maps text to 20- dimensional embedding vectors. There are several problems with carrying out inferences by calling model.predict() on an in-memory object (or a trainable object loaded into memory) as described in the preceding code snippet:

* We have to load the entire Keras model into memory. The text embedding layer, which was set up to be trainable, can be quite large because it needs to store embeddings for the full vocabulary of English words. Deep learning models with many layers can also be quite large. 
* The preceding architecture imposes limits on the latency that can be achieved because calls to the predict() method have to be sent one by one. 
* Even though the data scientist’s programming language of choice is Python, model inference is likely to be invoked by programs written by developers who prefer other languages, or on mobile platforms like Android or iOS that require different languages. 
* The model input and output that is most effective for training may not be user friendly. In our example, the model output was logits because it is better for gradient descent. This is why the second number in the output array is greater than 1. What clients will typically want is the sigmoid of this so that the output range is 0 to1 and can be interpreted in a more user-friendly format as a probability.

**Solution.** The solution consists of the following steps:

1. Export the model into a format that captures the mathematical core of the model and is programming language agnostic. Typically, the trained weight values are constants in the mathematical formula: `model.save('path')`

2. In the production system, the formula consisting of the “forward” calculations of the model is restored as a stateless function.  Here is how we can obtain the serving function and use it for inference:

   ```python
   serving_fn = tf.keras.models.load_model(export_path).signatures['serving_default']
   
   outputs = serving_fn(full_text_input= tf.constant([review1,review2, review3]))
   logit = outputs['positive_review_logits']
   ```

   The signature specifies that the prediction method takes a one-element array as input (called full_text_input) that is a string, and outputs one floating point number whose name is positive_review_logits. These names come from the names that we assigned to the Keras layers.

3. The stateless function is deployed into a framework that provides a REST endpoint. 

   The code above can be put into a web application or serverless framework such as Google App Engine, Heroku, AWS Lambda, Azure Functions, Google Cloud Functions, Cloud Run, and so on. What all these frameworks have in common is that they allow the developer to specify a function that needs to be executed. The frameworks take care of autoscaling the infrastructure so as to handle large numbers of prediction requests per second at low latency. For example, we can invoke the serving function from within Cloud Functions as follows:

   ```python
   serving_fn = None
   def handler(request):
       global serving_fn
       if serving_fn is None:
           serving_fn = (tf.keras.models.load_model(export_path).signatures['serving_default'])
           
       request_json = request.get_json(silent=True)
       if request_json and 'review' in request_json:
           review = request_json['review']
           outputs = serving_fn(full_text_input=tf.constant([review]))
           return outputs['positive_review_logits']
   ```

   Note that we should be careful to define the serving function as a global variable (or a singleton class) so that it isn’t reloaded in response to every request. In practice, the serving function will be reloaded from the export path (on Google Cloud Storage) only in the case of cold starts.

**Why it works.** The approach of exporting a model to a stateless function and deploying the stateless function in a web application framework works because web application frameworks offer autoscaling, can be fully managed, and are language neutral.

* Autoscaling - Scaling web endpoints to millions of requests per second is a wellunderstood engineering problem. Rather than building services unique to machine learning, we can rely on the decades of engineering work that has gone into building resilient web applications and web servers. 
* Fully managed - Cloud platforms abstract away the managing and installation of components like TensorFlow Serving as well. Thus, on Google Cloud, deploying the serving function as a REST API is very simple. With a REST endpoint in place, we can send a prediction request as a JSON.
* Language-neutral - Every modern programming language can speak REST, and a discovery service is provided to autogenerate the necessary HTTP stubs. Thus, Python clients can invoke the REST API as follows. Note that there is nothing framework specific in the code below. Because the cloud service abstracts the specifics of our ML model, we don’t need to provide any references to Keras or TensorFlow.
* Powerful ecosystem - Because web application frameworks are so widely used, there is a lot of tooling available to measure, monitor, and manage web applications. If we deploy the ML model to a web application framework, the model can be monitored and throttled using tools that software reliability engineers (SREs), IT administrators, and DevOps personnel are familiar with. They do not have to know anything about machine learning.

**Trade-Offs and Alternatives.** 

* Multiple signatures - It is quite common for models to support multiple objectives or clients who have different needs. While outputting a dictionary can allow different clients to pull out whatever they want, this may not be ideal in some cases. For example, the function we had to invoke to get a probability from the logits was simply tf.sigmoid(). This is pretty inexpensive, and there is no problem with computing it even for clients who will discard it. On the other hand, if the function had been expensive, computing it for clients who don’t need the value can add considerable overhead.

* Online prediction - Because the exported serving function is ultimately just a file format, it can be used to provide online prediction capabilities when the original machine learning training framework does not natively support online predictions.

* Prediction library - Instead of deploying the serving function as a microservice that can be invoked via a REST API, it is possible to implement the prediction code as a library function. The library function would load the exported model the first time it is called, invoke model.predict() with the provided input, and return the result. Application developers who need to predict with the library can then include the library with their applications.

  A library function is a better alternative than a microservice if the model cannot be called over a network either because of physical reasons (there is no network connectivity) or because of performance constraints. The library function approach also places the computational burden on the client, and this might be preferable from a budgetary standpoint. The main drawback of the library approach is that maintenance and updates of the model are difficult—all the client code that uses the model will have to be updated to use the new version of the library. 








### Batch Serving



The Batch Serving design pattern uses software infrastructure commonly used for distributed data processing to carry out inference on a large number of instances all at once.

**Problem.** Commonly, predictions are carried one at a time and on demand. Whether or not a credit card transaction is fraudulent is determined at the time a payment is being processed. The serving framework is architected to process an individual request synchronously and as quickly as possible, as discussed in Stateless Pattern. The serving infrastructure is usually designed as a microservice that offloads the heavy computation (such as with deep convolutional neural networks) to high-performance hardware such as tensor processing units (TPUs) or graphics processing units (GPUs) and minimizes the inefficiency associated with multiple software layers. 

However, there are circumstances where predictions need to be carried out asynchronously over large volumes of data. For example, determining whether to reorder a stock-keeping unit (SKU) might be an operation that is carried out hourly, not every time the SKU is bought at the cash register. Music services might create personalized daily playlists for every one of their users and push them out to those users.

**Solution.** The Batch Serving design pattern uses a distributed data processing infrastructure (MapReduce, Apache Spark, BigQuery, Apache Beam, and so on) to carry out ML inference on a large number of instances asynchronously.

**Trade-Offs and Alternatives.** The Batch Serving design pattern depends on the ability to split a task across multiple workers. So, it is not restricted to data warehouses or even to SQL. Any MapReduce framework will work. However, SQL data warehouses tend to be the easiest and are often the default choice, especially when the data is structured in nature.

* Cached results of batch serving - We discussed batch serving as a way to invoke a model over millions of items when the model is normally served online using the Stateless Serving Function design pattern. Of course, it is possible for batch serving to work even if the model does not support online serving. What matters is that the machine learning framework doing inference is capable of taking advantage of embarrassingly parallel processing.

* Lambda architecture - A production ML system that supports both online serving and batch serving is called a Lambda architecture—such a production ML system allows ML practitioners to trade-off between latency (via the Stateless Serving Function pattern) and throughput (via the Batch Serving pattern).

  Typically, a Lambda architecture is supported by having separate systems for online serving and batch serving. In Google Cloud, for example, the online serving infrastructure is provided by Cloud AI Platform Predictions and the batch serving infrastructure is provided by BigQuery and Cloud Dataflow.



### Continued Model Evaluation



The Continued Model Evaluation design pattern handles the common problem of needing to detect and take action when a deployed model is no longer fit-for-purpose.

**Problem.** The world is dynamic, but developing a machine learning model usually creates a static model from historical data. This means that once the model goes into production, it can start to degrade and its predictions can grow increasingly unreliable. Two of the main reasons models degrade over time are concept drift (relationship between the model inputs and target have changed) and data drift (any change that has occurred to the data being fed to your model for prediction as compared to the data that was used for training).

**Solution.** The most direct way to identify model deterioration is to continuously monitor your model’s predictive performance over time, by capture model predictions and true labels, and assess that performance with the same evaluation metrics you used during development.

**Trade-Offs and Alternatives.** The goal of continuous evaluation is to provide a means to monitor model performance and keep models in production fresh. In this way, continuous evaluation provides a trigger for when to retrain the model. In this case, it is important to consider tolerance thresholds for model performance, the tradeoffs they pose, and the role of scheduled retraining.

* Triggers for retraining - Depending on the complexity of the model and ETL pipelines, the cost of retraining could be expensive. The trade-off to consider is what amount of deterioration of performance is acceptable in relation to this cost.

  The threshold itself could be set as an absolute value; for example, model retraining occurs once model accuracy falls below 95%. Or the threshold could be set as a rate of change of performance, for example, once performance begins to experience a downward trajectory.

* Scheduled retraining - Where continued evaluation may happen every day, scheduled retraining jobs may occur only every week or every month. Once a new version of the model is trained, its performance is compared against the current model version. The updated model is deployed as a replacement only if it outperforms the previous model with respect to a test set of current data.

  In either case, it is helpful to have an automated pipeline set up that can execute the full retraining process with a single API call. Tools like Cloud Composer/Apache Airflow and AI Platform Pipelines are useful to create, schedule, and monitor ML workflows from preprocessing raw data and training to hyperparameter tuning and deployment.

* Data validation with TFX -  The Data Validation library can be used to compare the data examples used in training with those collected during serving. Validity checks detect anomalies in the data, training-serving skew, or data drift.





### Two-Phase Predictions



The Two-Phase Predictions design pattern provides a way to address the problem of keeping large, complex models performant when they have to be deployed on distributed devices by splitting the use cases into two phases, with only the simpler phase being carried out on the edge.

**Problem.** When deploying machine learning models, we cannot always rely on end users having reliable internet connections. In such situations, models are deployed at the edge—meaning they are loaded on a user’s device and don’t require an internet connection to generate predictions. Given device constraints, models deployed on the edge typically need to be smaller than models deployed in the cloud, and consequently require balancing trade-offs between model complexity and size, update frequency, accuracy, and low latency.

To convert a trained model into a format that works on edge devices, models often go through a process known as quantization, where learned model weights are represented with fewer bytes (example: TFLite). Quantization and other techniques employed by TF Lite significantly reduce the size and prediction latency of resulting ML models, but with that may come reduced model accuracy. Additionally, since we can’t consistently rely on edge devices having connectivity, deploying new model versions to these devices in a timely manner also presents a challenge. To account for these trade-offs, we need a solution that balances the reduced size and latency of edge models against the added sophistication and accuracy of cloud models.

**Solution.** With the Two-Phase Predictions design pattern, we split our problem into two parts. We start with a smaller, cheaper model that can be deployed on device. Because this model typically has a simpler task, it can accomplish this task on-device with relatively high accuracy. This is followed by a second, more complex model deployed in the cloud and triggered only when needed. Of course, this design pattern requires you to have a problem that can be split into two parts with varying levels of complexity.

One example of such a problem is smart devices like Google Home, which are activated by a wake word and can then answer questions and respond to commands related to setting alarms, reading the news, and interacting with integrated devices like lights and thermostats. Google Home, for example, is activated by saying “OK Google” or “Hey Google.” Once the device recognizes a wake word, users can ask more complex questions like, “Can you schedule a meeting with Sara at 10 a.m.?”

**Trade-Offs and Alternatives.** While the Two-Phase Predictions pattern works for many cases, there are situations where your end users may have very little internet connectivity and you therefore cannot rely on being able to call a cloud-hosted model.

* Standalone single-phase model - Even though these users’ devices won’t be able to reliably access a cloud model, it’s still important to give them a way to access your application. For this case, rather than relying on a two-phase prediction flow, you can make your first model robust enough that it can be self-sufficient. To do this, we can create a smaller version of our complex model, and give users the option to download this simpler, smaller model for use when they are offline.

* Offline support for specific use cases - Another solution for making your application work for users with minimal internet connectivity is to make only certain parts of your app available offline. 

* Handling many predictions in near real time - In other cases, end users of your ML model may have reliable connectivity but might need to make hundreds or even thousands of predictions to your model at once. If you only have a cloud-hosted model and each prediction requires an API call to a hosted service, getting prediction responses on thousands of examples at once will take too much time.

  To understand this, let’s say we have embedded devices deployed in various areas throughout a user’s house. These devices are capturing data on temperature, air pressure, and air quality. We have a model deployed in the cloud for detecting anomalies from this sensor data. Because the sensors are continuously collecting new data, it would be inefficient and expensive to send every incoming data point to our cloud model. Instead, we can have a model deployed directly on the sensors to identify possible anomaly candidates from incoming data. We can then send only the potential anomalies to our cloud model for consolidated verification, taking sensor readings from all the locations into account.





### Keyed Predictions



Normally, you train your model on the same set of input features that the model will be supplied in real time when it is deployed. In many situations, however, it can be advantageous for your model to also pass through a client supplied key. This is called the Keyed Predictions design pattern.

**Problem.** If your model is deployed as a web service and accepts a single input, then it is quite clear which output corresponds to which input. But what if your model accepts a file with a million inputs and sends back a file with a million output predictions?

You might think that it should be obvious that the first output instance corresponds to the first input instance, the second output instance to the second input instance, etc. However, with a 1:1 relationship, it is necessary for each server node to process the full set of inputs serially. It would be much more advantageous if you use a distributed data processing system and farm out instances to multiple machines, collect all the resulting outputs, and send them back. The problem with this approach is that the outputs are going to be jumbled. Requiring that the outputs be ordered the same way poses scalability challenges, and providing the outputs in an unordered manner requires the clients to somehow know which output corresponds to which input.

**Solution.** The solution is to use pass-through keys. Have the client supply a key associated with each input. For example, suppose your model is trained with three inputs (a, b, c) to produce the output d. Make your clients supply (k, a, b, c) to your model where k is a key with a unique identifier. The key could be as simple as numbering the input instances 1, 2, 3, …, etc. Your model will then return (k, d), and so the client will be able to figure out which output instance corresponds to which input instance.

**Trade-Offs and Alternatives.** 

* Asynchronous serving - Many production machine learning models these days are neural networks, and neural networks involve matrix multiplications. Matrix multiplication on hardware like GPUs and TPUs is more efficient if you can ensure that the matrices are within certain size ranges and/or multiples of a certain number. It can, therefore, be helpful to accumulate requests (up to a maximum latency of course) and handle the incoming requests in chunks. Since the chunks will consist of interleaved requests from multiple clients, the key, in this case, needs to have some sort of client identifier as well.
* Continuous evaluation - If you are doing continuous evaluation, it can be helpful to log metadata about the prediction requests so that you can monitor whether performance drops across the board, or only in specific situations. Such slicing is made much easier if the key identifies the situation in question. For example, suppose that we need to apply a Fairness Lens to ensure that our model’s performance is fair across different customer segments (age of customer and/or race of customer, for example). The model will not use the customer segment as an input, but we need to evaluate the performance of the model sliced by the customer segment. In such cases, having the customer segment(s) be embedded in the key makes slicing easier.





##  Reproducibility Design Patterns



Software best practices such as unit testing assume that if we run a piece of code, it produces deterministic output. This sort of reproducibility is difficult in machine learning. During training, machine learning models are initialized with random values and then adjusted based on training data. Beyond the random seed, there are many other artifacts that need to be fixed in order to ensure reproducibility during training. In addition, machine learning consists of different stages, such as training, deployment, and retraining. It is often important that some things are reproducible across these stages as well.



### Transform



The Transform design pattern makes moving an ML model to production much easier by keeping inputs, features, and transforms carefully separate.

**Problem.** The problem is that the inputs to a machine learning model are not the features that the machine learning model uses in its computations. In a text classification model, for example, the inputs are the raw text documents and the features are the numerical embedding representations of this text. Note that, at inference time, we have to know what features the model was trained on, how they should be interpreted, and the details of the transformations that were applied. Training-serving skew, caused by differences in any of factors between the training and serving environments, is one of the key reasons why productionization of ML models is so hard.

**Solution.** The solution is to explicitly capture the transformations applied to convert the model inputs into features. 

**Trade-Offs and Alternatives.** 

* Transformations in TensorFlow and Keras - First, make every input to the Keras model an Input layer. Second, maintain a dictionary of transformed features, and make every transformation either a Keras Preprocessing layer or a Lambda layer. Third, all these transformed layers will be concatenated into a DenseFeatures layer.
* Efficient transformations with tf.transform - One drawback to the above approach is that the transformations will be carried out during each iteration of training. The tf.transform library (which is part of TensorFlow Extended) provides an efficient way of carrying out transformations over a preprocessing pass through the data and saving the resulting features and transformation artifacts so that the transformations can be applied by TensorFlow Serving during prediction time.
* Alternate pattern approaches - An alternative approach to solving the training-serving skew problem is to employ the Feature Store pattern. The feature store comprises a coordinated computation engine and repository of transformed feature data. The computation engine supports low-latency access for inference and batch creation of transformed features while the data repository provides quick access to transformed features for model training. The advantage of a feature store is there is no requirement for the transformation operations to fit into the model graph. For example, as long as the feature store supports Java, the preprocessing operations could be carried out in Java while the model itself could be written in PyTorch. The disadvantage of a feature store is that it makes the model dependent on the feature store and makes the serving infrastructure much more complex.





###  Repeatable Splitting



To ensure that sampling is repeatable and reproducible, it is necessary to use a well distributed column and a deterministic hash function to split the available data into training, validation, and test datasets.

**Problem.** Many machine learning tutorials will suggest splitting data randomly into training, validation, and test datasets. Unfortunately, this approach fails in many real-world situations. The reason is that it is rare that the rows are independent. For example, if we are training a model to predict flight delays, the arrival delays of flights on the same day will be highly correlated with one another. This leads to leakage of information between the training and testing dataset when some of the flights on any particular day are in the training dataset and some other flights on the same day are in the testing dataset.

**Solution.** First, we identify a column that captures the correlation relationship between rows. In our airline delay dataset, this is the date column. Then, we use the last few digits of a hash function on that column to split the data. For the airline delay problem, we can use the Farm Fingerprint hashing algorithm on the date column to split the available data into training, validation, and testing datasets.

**Trade-Offs and Alternatives.** 

* Random split - What if the rows are not correlated? In that case, we want a random, repeatable split but do not have a natural column to split by. We can hash the entire row of data by converting it to a string and hashing that string.
* Split on multiple columns - simply concatenate the fields (this is a feature cross) before computing the hash.
* Sequential split - In the case of time-series models, a common approach is to use sequential splits of data.



### Bridged Schema



The Bridged Schema design pattern provides ways to adapt the data used to train a model from its older, original data schema to newer, better data. This pattern is useful because when an input provider makes improvements to their data feed, it often takes time for enough data of the improved schema to be collected for us to adequately train a replacement model. The Bridged Schema pattern allows us to use as much of the newer data as is available, but augment it with some of the older data to improve model accuracy.

**Problem.** Consider a point-of-sale application that suggests how much to tip a delivery person. The application might use a machine learning model that predicts the tip amount, taking into account the order amount, delivery time, delivery distance, and so on. Such a model would be trained on the actual tips added by customers. Assume that one of the inputs to the model is the payment type. In the historical data, this has been recorded as “cash” or “card.” However, let’s say the payment system has been upgraded and it now provides more detail on the type of card (gift card, debit card, credit card) that was used. At prediction time, the newer information will always be available since we are always predicting tip amounts on transactions conducted after the payment system upgrade. Because the new information is extremely valuable, and it is already available in production to the prediction system, we would like to use it in the model as soon as possible.

**Solution.** The solution is to bridge the schema of the old data to match the new data. Then, we train an ML model using as much of the new data as is available and augment it with the older data. There are two questions to answer. First, how will we square the fact that the older data has only two categories for payment type, whereas the new data has four categories? Second, how will the augmentation be done to create datasets for training, validation, and testing?

What we do know is that a transaction coded as “card” in the old data would have been one of these types but the actual type was not recorded. It’s possible to bridge the schema probabilistically or statically. 

* Probabilistically - Imagine that we estimate from the newer training data that of the card transactions, 10% are gift cards, 30% are debit cards, and 60% are credit cards. Each time an older training example is loaded into the trainer program, we could choose the card type by generating a uniformly distributed random number in the range [0, 100) and choosing a gift card when the random number is less than 10, a debit card if it is in [10, 40), and a credit card otherwise.

* Statically - Categorical variables are usually one-hot encoded. If we follow the probabilistic approach above and train long enough, the average one-hot encoded value presented to the training program of a “card” in the older data will be [0, 0.1, 0.3, 0.6]. The first 0 corresponds to the cash category. The second number is 0.1 because 10% of the time, on card transactions, this number will be 1 and it will be zero in all other cases. Similarly, we have 0.3 for debit cards and 0.6 for credit cards. To bridge the older data into the newer schema, we can transform the older categorical data into this representation where we insert the a priori probability of the new classes as estimated from the training data. The newer data, on the other hand, will have [0, 0, 1, 0] for a transaction that is known to have been paid by a debit card.



**Trade-Offs and Alternatives.** 

* Union schema - It can be tempting to simply create a union of the older and newer schemas. For example, we could define the schema for the payment type as having five possible values: cash, card, gift card, debit card, and credit card. The backward-compatible, union-of-schemas approach doesn’t work for machine learning though.
* Handling new features - If we have new input features we want to start using immediately, we should bridge the older data (where this new feature will be missing) by imputing a value for the new feature. Recommended choices for the imputation value are: 
  * The mean value of the feature if the feature is numeric and normally distributed 
  * The median value of the feature if the feature is numeric and skewed or has lots of outliers 
  * The median value of the feature if the feature is categorical and sortable 
  * The mode of the feature if the feature is categorical and not sortable 
  * The frequency of the feature being true if it is boolean

* Handling precision increases - When the input provider increases the precision of their data stream, follow the bridging approach to create a training dataset that consists of the higher-resolution data, augmented with some of the older data.





### Windowed Inference



The Windowed Inference design pattern handles models that require an ongoing sequence of instances in order to run inference. This pattern works by externalizing the model state and invoking the model from a stream analytics pipeline. This pattern is also useful when a machine learning model requires features that need to be computed from aggregates over time windows. By externalizing the state to a stream pipeline, the Windowed Inference design pattern ensures that features calculated in a dynamic, time-dependent way can be correctly repeated between training and serving. It is a way of avoiding training–serving skew in the case of temporal aggregate features.

**Problem.** Take a look at the arrival delays at Dallas Fort Worth (DFW) airport depicted for a couple of days. Early in the morning, most flights are on time, so even the small spike is anomalous. By the middle of the day (after 12 p.m. on May 10), variability picks up and 25-minute delays are quite common, but a 75-minute delay is still unusual. Whether or not a specific delay is anomalous depends on a time context, for example, on the arrival delays observed over the past two hours.

Given that the flight above (at 08:45 on February 3) is 19 minutes late, is that unusual or not? Commonly, to carry out ML inference on a flight, we only need the features of that flight. In this case, however, the model requires information about all flights to DFW airport between 06:45 and 08:45. It is not possible to carry out inference one flight at a time. We need to somehow provide the model information about all the previous flights.

**Solution.** The solution is to carry out stateful stream processing—that is, stream processing that keeps track of the model state through time:

* A sliding window is applied to flight arrival data.
* The internal model state (this could be the list of flights) is updated with flight information every time a new flight arrives, thus building a 2-hour historical record of flight data
* Every time the window is closed (every 10 minutes for example), a time-series ML model is trained on the 2-hour list of flights. This model is then used to predict future flight delays and the confidence bounds of such predictions.
* The time-series model parameters are externalized into a state variable. We could use a time-series model such as autoregressive integrated moving average (ARIMA) or long short-term memory (LSTMs), in which case, the model parameters would be the ARIMA model coefficients or the LSTM model weights. To keep the code understandable, we will use a zero-order regression model, and so our model parameters will be the average flight delay and the variance of the flight delays over the two-hour window.
* When a flight arrives, its arrival delay can be classified as anomalous or not using the externalized model state—it is not necessary to have the full list of flights over the past 2 hours.





### Workflow Pipeline

In the Workflow Pipeline design pattern, we address the problem of creating an end-to-end reproducible pipeline by containerizing and orchestrating the steps in our machine learning process. The containerization might be done explicitly, or using a framework that simplifies the process.

**Problem.** An individual data scientist may be able to run data preprocessing, training, and model deployment steps from end to end within a single script or notebook. However, as each step in an ML process becomes more complex, and more people within an organization want to contribute to this code base, running these steps from a single notebook will not scale.

In traditional programming, monolithic applications are described as those where all of the application’s logic is handled by a single program. To test a small feature in a monolithic app, we must run the entire program. The same goes for deploying or debugging monolithic applications. Deploying a small bug fix for one piece of the program requires deploying the entire application, which can quickly become unwieldy. When the entire codebase is inextricably linked, it becomes difficult for individual developers to debug errors and work independently on different parts of the application. In recent years, monolithic apps have been replaced in favor of a microservices architecture where individual pieces of business logic are built and deployed as isolated (micro) packages of code. With microservices, a large application is split into smaller, more manageable parts so that developers can build, debug, and deploy pieces of an application independently.

**Solution.** To handle the problems that come with scaling machine learning processes, we can make each step in our ML workflow a separate, containerized service. Containers guarantee that we’ll be able to run the same code in different environments, and that we’ll see consistent behavior between runs. These individual containerized steps are then chained together to make a pipeline that can be run with a REST API call. 

Because pipeline steps run in containers, we can run them on a development laptop, with on-premises infrastructure, or with a hosted cloud service. This pipeline workflow allows team members to build out pipeline steps independently. Containers also provide a reproducible way to run an entire pipeline end to end, since they guarantee consistency among library dependency versions and runtime environments. Additionally, because containerizing pipeline steps allows for a separation of concerns, individual steps can use different runtimes and language versions.

There are many tools for creating pipelines with both on-premise and cloud options available, including Cloud AI Platform Pipelines, TensorFlow Extended (TFX), Kubeflow Pipelines (KFP), MLflow, and Apache Airflow. 

**Pipeline steps.** ML pipeline examples: https://github.com/Building-ML-Pipelines/building-machine-learning-pipelines

The pipeline should include steps that: 

* Version your data effectively and kick off a new model training run - In this pipeline step, we process the data into a format that the following components can digest. The data ingestion step does not perform any feature engineering.

* Validate the received data and check against data drift - Data validation focuses on checking that the statistics of the new data are as expected (e.g., the range, number of categories, and distribution of categories). It also alerts the data scientist if any anomalies are detected. For example, if you are training a binary classification model, your training data could contain 50% of Class X samples and 50% of Class Y samples. Data validation tools provide alerts if the split between these classes changes, where perhaps the newly collected data is split 70/30 between the two classes.

* Efficiently preprocess data for your model training and validation 

* Effectively train your machine learning models 

* Track your model training 

* Analyze and validate your trained and tuned models - Generally, we would use accuracy or loss to determine the optimal set of model parameters. But once we have settled on the final version of the model, it’s extremely useful to carry out a more in-depth analysis of the model’s performance. This may include calculating other metrics such as precision, recall, and AUC (area under the curve), or calculating performance on a larger dataset than the validation set used in training.

* Model versioning - The purpose of the model versioning and validation step is to keep track of which model, set of hyperparameters, and datasets have been selected as the next version to be deployed.

* Deploy the validated model - Once you have trained, tuned, and analyzed your model, it is ready for prime time. Unfortunately, too many models are deployed with one-off implementations, which makes updating models a brittle process. Modern model servers (such as TF Serving or some cloud solution) allow you to deploy your models without writing web app code. Often, they provide multiple API interfaces like representational state transfer (REST) or remote procedure call (RPC) protocols and allow you to host multiple versions of the same model simultaneously. Hosting multiple versions at the same time will allow you to run A/B tests on your models and provide valuable feedback about your model improvements.

  Deployment approaches:

  1. Server-Side Deployment - Server-side deployment consists of setting up a web server that can accept requests from clients, run them through an inference pipeline, and return the results. This solution fits within a web development paradigm, as it treats models as another endpoint in an application. There are two common workloads for server-side models, streaming and batch. Streaming workflows accept requests as they come and process them immediately. Batch workflows are run less frequently and process a large number of requests all at once.
  2. Client-Side Deployment - The goal of deploying models on the client side is to run all computations on the client, eliminating the need for a server to run models. Computers, tablets, modern smartphones, and some connected devices such as smart speakers or doorbells have enough computing power to run models themselves. 

  Deploy model with TF Server example: https://www.tensorflow.org/tfx/tutorials/serving/rest_simple and https://github.com/sthalles/deeplab_v3 (https://medium.com/free-code-camp/how-to-deploy-tensorflow-models-to-production-using-tf-serving-4b4b78d41700)

* Scale the deployed model - So far, we have discussed the deployment of a single TensorFlow Serving instance hosting one or more model versions. While this solution is sufficient for a good number of deployments, it isn’t enough for applications experiencing a high volume of prediction requests. In these situations, your single Docker container with TensorFlow Serving needs to be replicated to reply to the additional prediction requests. The orchestration of the container replication is usually managed by tools like Docker Swarm or Kubernetes. 

* Capture new training data and model performance metrics with feedback loops - The last step of the machine learning pipeline is often forgotten, but it is crucial to the success of data science projects. We can then measure the effectiveness and performance of the newly deployed model. During this step, we can capture valuable information about the performance of the model. In some situations, we can also capture new training data to increase our datasets and update our model.



The pipeline is actually a recurring cycle. Data can be continuously collected and, therefore, machine learning models can be updated. More data generally means improved models. And because of this constant influx of data, automation is key. In real-world applications, you want to retrain your models frequently. If you don’t, in many cases accuracy will decrease because the training data is different from the new data that the model is making predictions on. If retraining is a manual process, where it is necessary to manually validate the new training data or analyze the updated models, a data scientist or machine learning engineer would have no time to develop new models for entirely different business problems.



![](https://drive.google.com/uc?export=view&id=13dIUvNRYnB7YdEVJf8bE2PAsz6sPiXS4)

**TensorFlow Extended.** Machine learning pipelines can become very complicated and consume a lot of overhead to manage task dependencies. At the same time, machine learning pipelines can include a variety of tasks, including tasks for data validation, preprocessing, model training, and any post-training tasks. Having brittle connections ultimately means that production models will be updated infrequently, and data scientists and machine learning engineers loathe updating stale models. Google faced the same problem internally and decided to develop a platform to simplify the pipeline definitions and to minimize the amount of task boilerplate code to write. The open source version of Google’s internal ML pipeline framework is TFX.



Figure 2-2 shows the general pipeline architecture with TFX. Pipeline orchestration tools are the foundation for executing our tasks. Besides the orchestration tools, we need a data store to keep track of the intermediate pipeline results. The individual components communicate with the data store to receive their inputs, and they return the results to the data store. These results can then be inputs to following tasks. TFX provides the layer that combines all of these tools, and it provides the individual components for the major pipeline tasks.

![](https://drive.google.com/uc?export=view&id=1_61G8vU4C2LKY1ZBHOXnZP8rFOVxhjwi)

TFX provides a variety of pipeline components that cover a good number of use cases:

![](https://drive.google.com/uc?export=view&id=11UmOJhKOtpnTF9d6FzKyIfwvlOhxrr9z)



All machine learning pipeline components read from a channel to get input artifacts from the metadata store. The data is then loaded from the path provided by the metadata store and processed. The output of the component, the processed data, is then provided to the next pipeline components.

In TFX terms, the three internal parts of the component are called the driver, executor, and publisher. The driver and the publisher aren’t moving any data. Instead, they read and write references from the Metadata Store. The inputs and outputs of the components are called artifacts.

One advantage of passing the metadata between components instead of the direct artifacts is that the information can be centrally stored.

In practice, the workflow goes as follows: when we execute a component, it uses the ML Metadata (MLMD) API to save the metadata corresponding to the run. For example, the component driver receives the reference for a raw dataset from the metadata store. After the component execution, the component publisher will store the references of the component outputs in the metadata store. MLMD saves the metadata consistently to a MetadataStore, based on a storage backend. Currently, MLMD supports three types of backends: 

* In-memory database (via SQLite) 
* SQLite 
* MySQL

A variety of TFX components and libraries (e.g., TensorFlow Transform) rely on Apache Beam to process pipeline data efficiently. Since it is incredibly versatile, Apache Beam can be used to describe batch processes, streaming operations, and data pipelines.





### Feature Store



The Feature Store design pattern simplifies the management and reuse of features across projects by decoupling the feature creation process from the development of models using those features.

**Problem.** In short, the ad hoc approach to feature engineering slows model development and leads to duplicated effort and work stream inefficiency. Furthermore, feature creation is inconsistent between training and inference, running the risk of training–serving skew or data leakage by accidentally introducing label information into the model input pipeline.

**Solution.** The solution is to create a shared feature store, a centralized location to store and document feature datasets that will be used in building machine learning models and can be shared across projects and teams. The feature store acts as the interface between the data engineer’s pipelines for feature creation and the data scientist’s workflow building models using those features. This also allows the basic software engineering principles of versioning, documentation, and access control to be applied to the features that are created.

A typical feature store is built with two key design characteristics: tooling to process large feature data sets quickly, and a way to store features that supports both low-latency access (for inference) and large batch access (for model training).



### Model Versioning

In the Model Versioning design pattern, backward compatibility is achieved by deploying a changed model as a microservice with a different REST endpoint. This is a necessary prerequisite for many of the other patterns discussed in this chapter.

**Problem.** As we’ve seen with data drift, models can become stale over time and need to be updated regularly to make sure they reflect an organization’s changing goals, and the environment associated with their training data. Deploying model updates to production will inevitably affect the way models behave on new data, which presents a challenge—we need an approach for keeping production models up to date while still ensuring backward compatibility for existing model users.

**Solution.** To gracefully handle updates to a model, deploy multiple model versions with different REST endpoints. This ensures backward compatibility—by keeping multiple versions of a model deployed at a given time, those users relying on older versions will still be able to use the service. Versioning also allows for fine-grained performance monitoring and analytics tracking across versions. We can compare accuracy and usage statistics, and use this to determine when to take a particular version offline. If we have a model update that we want to test with only a small subset of users, the Model Versioning design pattern makes it possible to perform A/B testing.





## Responsible AI



### Heuristic Benchmark



The Heuristic Benchmark pattern compares an ML model against a simple, easy-to-understand heuristic in order to explain the model’s performance to business decision makers.

**Problem.** Suppose a bicycle rental agency wishes to use the expected duration of rentals to build a dynamic pricing solution. After training an ML model to predict the duration of a bicycle’s rental period, they evaluate the model on a test dataset and determine that the mean absolute error (MAE) of the trained ML model is 1,200 seconds. When they present this model to the business decision makers, they will likely be asked: “Is an MAE of 1,200 seconds good or bad?” This is a question we need to be ready to handle whenever we develop a model and present it to business stakeholders.

**Solution.** If this is the second ML model being developed for a task, an easy answer is to compare the model’s performance against the currently operational version. It is quite easy to say that the MAE is now 30 seconds lower. 

But what if there is no current production methodology in place, and we are building the very first model for a green-field task? In such cases, the solution is to create a simple benchmark for the sole purpose of comparing against our newly developed ML model. We call this a heuristic benchmark.

A good heuristic benchmark should be intuitively easy to understand and relatively trivial to compute. If we find ourselves defending or debugging the algorithm used by the benchmark, we should search for a simpler, more understandable one. Good examples of a heuristic benchmark are constants, rules of thumb, or bulk statistics (such as the mean, median, or mode).



### Explainable Predictions



The Explainable Predictions design pattern increases user trust in ML systems by providing users with an understanding of how and why models make certain predictions. While models such as decision trees are interpretable by design, the architecture of deep neural networks makes them inherently difficult to explain. For all models, it is useful to be able to interpret predictions in order to understand the combinations of features influencing model behavior.

**Problem.** When evaluating a machine learning model to determine whether it’s ready for production, metrics like accuracy, precision, recall, and mean squared error only tell one piece of the story. They provide data on how correct a model’s predictions are relative to ground truth values in the test set, but they carry no insight on why a model arrived at those predictions. In many ML scenarios, users may be hesitant to accept a model’s prediction at face value.

Finally, as data scientists and ML engineers, we can only improve our model quality to a certain degree without an understanding of the features it’s relying on to make predictions. We need a way to verify that models are performing in the way we expect. For example, let’s say we are training a model on tabular data to predict whether a flight will be delayed. The model is trained on 20 features. Under the hood, maybe it’s relying only on 2 of those 20 features, and if we removed the rest, we could significantly improve our system’s performance. Or maybe each of those 20 features is necessary to achieve the degree of accuracy we need. Without more details on what the model is using, it’s difficult to know.

**Solution.** Techniques for understanding and communicating how and why an ML model makes predictions is an area of active research. Also called interpretability or model understanding, explainability is a new and rapidly evolving field within ML, and can take a variety of forms depending on a model’s architecture and the type of data it is trained on.





## Machine Learning techniques and tasks





### Active learning

Active learning is an interesting supervised learning paradigm. It is usually applied when obtaining labeled examples is costly. That is often the case in the medical or financial domains, where the opinion of an expert may be required to annotate patients’ or customers’ data. The idea is to start learning with relatively few labeled examples, and a large number of unlabeled ones, and then label only those examples that contribute the most to the model quality.

There are multiple strategies of active learning. Here, we discuss only the following two: 

1. data density and uncertainty based
2. support vector-based.

The former strategy applies the current model f, trained using the existing labeled examples, to each of the remaining unlabelled examples (or, to save the computing time, to some random sample of them). For each unlabeled example x, the following importance score is computed: density(x) · uncertaintyf (x). Density reflects how many examples surround x in its close neighborhood, while uncertaintyf (x) reflects how uncertain the prediction of the model f is for x. In binary classification with sigmoid, the closer the prediction score is to 0.5, the more uncertain is the prediction. In SVM, the closer the example is to the decision boundary, the most uncertain is the prediction

In multiclass classification, entropy can be used as a typical measure of uncertainty:
$$
H_f(x) = -\sum_{c=1}^C Pr(y^{(c)}; f(x))ln[Pr(y^{(c)}; f(x))]
$$
Density for the example x can be obtained by taking the average of the distance from x to each of its k nearest neighbors (with k being a hyperparameter).

Once we know the importance score of each unlabeled example, we pick the one with the highest importance score and ask the expert to annotate it. Then we add the new annotated example to the training set, rebuild the model and continue the process until some stopping criterion is satisfied. A stopping criterion can be chosen in advance (the maximum number of requests to the expert based on the available budget) or depend on how well our model performs according to some metric.

The support vector-based active learning strategy consists in building an SVM model using the labeled data. We then ask our expert to annotate the unlabeled example that lies the closest to the hyperplane that separates the two classes. The idea is that if the example lies closest to the hyperplane, then it is the least certain and would contribute the most to the reduction of possible places where the true (the one we look for) hyperplane could lie.

### Semi-supervised learning

In semi-supervised learning (SSL) we also have labeled a small fraction of the dataset; most of the remaining examples are unlabeled. Our goal is to leverage a large number of unlabeled examples to improve the model performance without asking for additional labeled examples.

The neural network architecture that attained a remarkable performance is called a **ladder network**. To understand ladder networks you have to understand what an **autoencoder** is.

An autoencoder is a feed-forward neural network with an encoder-decoder architecture. It is trained to reconstruct its input. So the training example is a pair (x, x). We want the output xˆ of the model f(x) to be as similar to the input x as possible.

An important detail here is that an autoencoder’s network looks like an hourglass with a bottleneck layer in the middle that contains the embedding of the D-dimensional input vector; the embedding layer usually has much fewer units than D. The goal of the decoder is to reconstruct the input feature vector from this embedding.

A **denoising autoencoder** corrupts the left-hand side x in the training example (x, x) by adding some random perturbation to the features. If our examples are grayscale images with pixels represented as values between 0 and 1, usually a normal Gaussian noise is added to each feature. 

A **ladder network** is a denoising autoencoder with an upgrade. The encoder and the decoder have the same number of layers. The bottleneck layer is used directly to predict the label (using the softmax activation function). The network has several cost functions. For each layer l of the encoder and the corresponding layer l of the decoder, one cost Cl d penalizes the difference between the outputs of the two layers (using the squared Euclidean distance). When a labeled example is used during training, another cost function, Cc, penalizes the error in prediction of the label (the negative log-likelihood cost function is used).

Other semi-supervised learning techniques, not related to training neural networks, exist. One of them implies building the model using the labeled data and then cluster the unlabeled and labeled examples together using any clustering technique. For each new example, we then output as a prediction the majority label in the cluster it belongs to. Another technique, called S3VM, is based on using SVM. We build one SVM model for each possible labeling of unlabeled examples and then we pick the model with the largest margin. The paper on S3VM describes an approach that allows solving this problem without actually enumerating all possible labelings.

### One-shot learning

In one-shot learning, typically applied in face recognition, we want to build a model that can recognize that two photos of the same person represent that same person. If we present to the model two photos of two different people, we expect the model to recognize that the two people are different. 

To solve such a problem, we could go a traditional way and build a binary classifier that takes two images as input and predict either true (when the two pictures represent the same person) or false (when the two pictures belong to different people). However, in practice, this would result in a neural network twice as big as a typical neural network, because each of the two pictures needs its own embedding subnetwork. Training such a network would be challenging not only because of its size but also because the positive examples would be much harder to obtain than negative ones. So the problem is highly imbalanced.

One way to effectively solve the problem is to train a siamese neural network (SNN). An SNN can be implemented as any kind of neural network, a CNN, an RNN, or an MLP. The network only takes one image as input at a time; so the size of the network is not doubled. To obtain a binary classifier “same_person”/“not_same” out of a network that only takes one picture as input, we train the networks in a special way. To train an SNN, we use the triplet loss function. For example, let us have three images of a face: image A (for anchor), image P (for positive) and image N (for negative). A and P are two different pictures of the same person; N is a picture of another person. Each training example i is now a triplet (Ai, Pi, Ni).

Let’s say we have a neural network model f that can take a picture of a face as input and output an embedding of this picture. The triplet loss for example i is defined as,
$$
max(||f(A_i) - f(P_i)||^2 - ||f(A_i) - f(N_i)||^2 + \alpha, 0)
$$
The cost function is defined as the average triplet loss:
$$
\frac{1}{N} \sum_{i=1}^N max(||f(A_i) -f(P_i)||^2 - ||f(A_i) - f(N_i)||^2 + \alpha, 0)
$$
where – is a positive hyperparameter. Intuitively, ||f(A) - f(P)||^2 is low when our neural network outputs similar embedding vectors for A and P; ||f(A) - f(N)||^2 is high when the embedding for pictures of two different people are different.



Rather than randomly choosing an image for N, a better way to create triplets for training is to use the current model after several epochs of learning and find candidates for N that are similar to A and P according to that model. Using random examples as N would significantly slow down the training because the neural network will easily see the difference between pictures of two random people, so the average triplet loss will be low most of the time and the parameters will not be updated fast enough. 

To build an SNN, we first decide on the architecture of our neural network. For example, CNN is a typical choice if our inputs are images. Given an example, to calculate the average triplet loss, we apply, consecutively, the model to A, then to P, then to N, and then we compute the loss for that example. We repeat that for all triplets in the batch and then compute the cost; gradient descent with backpropagation propagates the cost through the network to update its parameters.

### Zero-shot learning

In zero-shot learning (ZSL) we want to train a model to assign labels to objects. The most frequent application is to learn to assign labels to images.

However, contrary to standard classification, we want the model to be able to predict labels that we didn’t have in the training data. How is that possible? The trick is to use embeddings not just to represent the input x but also to represent the output y. Imagine that we have a model that for any word in English can generate an embedding vector with the following property: if a word yi has a similar meaning to the word yk, then the embedding vectors for these two words will be similar.

For example, if yi is Paris and yk is Rome, then they will have embeddings that are similar; on the other hand, if yk is potato, then the embeddings of yi and yk will be dissimilar. Such embedding vectors are called word embeddings, and they are usually compared using cosine similarity metrics.

Now, in our classification problem, we can replace the label yi for each example i in our training set with its word embedding and train a multi-label model that predicts word embeddings. To get the label for a new example x, we apply our model f to x, get the embedding yˆ and then search among all English words those whose embeddings are the most similar to yˆ using cosine similarity.



### Learning to Rank

Learning to rank is a supervised learning problem. Among others, one frequent problem solved using learning to rank is the optimization of search results returned by a search engine for a query. In search result ranking optimization, a labeled example Xi in the training set of size N is a ranked collection of documents of size ri (labels are ranks of documents). A feature vector represents each document in the collection. The goal of the learning is to find a ranking function f which outputs values that can be used to rank documents. For each training example, an ideal function f would output values that induce the same ranking of documents as given by the labels.

### Learning to Recommend

Learning to recommend is an approach to build recommender systems. Usually, we have a user who consumes content. We have the history of consumption and want to suggest this user new content that they would like. It could be a movie on Netflix or a book on Amazon. Traditionally, two approaches were used to give recommendations: content-based filtering and collaborative filtering.

Content-based filtering consists of learning what users like based on the description of the content they consume. For example, if the user of a news site often reads news articles on science and technology, then we would suggest to this user more documents on science and technology. More generally, we could create one training set per user and add news articles to this dataset as a feature vector x and whether the user recently read this news article as a label y. Then we build the model of each user and can regularly examine each new piece of content to determine whether a specific user would read it or not. The content-based approach has many limitations. For example, the user can be trapped in the so-called filter bubble: the system will always suggest to that user the information that looks very similar to what user already consumed. That could result in complete isolation of the user from information that disagrees with their viewpoints or expands them. On a more practical side, the users might just stop following recommendations, which is undesirable. Collaborative filtering has a significant advantage over content-based filtering: the recommendations to one user are computed based on what other users consume or rate. For instance, if two users gave high ratings to the same ten movies, then it’s more likely that user 1 will appreciate new movies recommended based on the tastes of the user 2 and vice versa. The drawback of this approach is that the content of the recommended items is ignored. In collaborative filtering, the information on user preferences is organized in a matrix. Each row corresponds to a user, and each column corresponds to a piece of content that user rated or consumed. Usually, this matrix is huge and extremely sparse, which means that most of its cells aren’t filled (or filled with a zero). The reason for such a sparsity is that most users consume or rate just a tiny fraction of available content items. It’s is very hard to make meaningful recommendations based on such sparse data.

Two effective recommender system learning algorithms are 

1. Denoising autoencoders (DAE).

2. Factorization machines (FM) . The factorization machine model is defined as follows:
   $$
   f(x) = b + \sum_{i=1}^D w_ix_i + \sum_{i=1}^D\sum_{j=i+1}^D(v_iv_j)x_ix_j
   $$
   Depending on the problem, the loss function could be squared error loss (for regression) or hinge loss. For classification with y œ {≠1, +1}, with hinge loss or logistic loss the prediction is made as y = sign(f(x)). The logistic loss is defined as,
   $$
   loss(f(x), y) = \frac{1}{ln2}ln(1 + e^{-yf(x)})
   $$





Вы должны взять матрицу X, заполнить все пустые ячейки средними значениями рейтинга для данного элемента (не нужно заполнять его нулями, поскольку это может означать нечто в рейтинговой системе, а SVD не может обрабатывать отсутствующие значения), а затем вычислить SVD. Теперь, когда вы произвели такое разложение, это значит, что вы захватили скрытые характеристики, которые при желании можете применять для сравнения пользователей. Но вам нужно не это — вы хотите предсказать. Перемножив U, S и Vτ , вы получите приближение A к X или предсказание , таким образом, можете прогнозировать оценку, просто просматривая запись для соответствующей пары «пользователь/ элемент» в матрице X. 





### Прогнозирование временных рядов

**Ресурсы:** Hyndman R.J., Athanasoupouls G. - Forecasting: principles and practice

Эконометрика - основной источник задач прогнозирования.

Основные явления в эконометрических временных рядах:

1. Тренд - главное долгосрочное изменение уровня ряда
2. Сезонности - циклическое изменение уровня ряда с постоянным периодом
3. Разладки (смены модели ряда)
4. Цикл - изменение уровня ряда с переменным периодом (экономические циклы, периоды солнечной активности)
5. Ошибка - непрогнозируемая случайная компонента ряда

В роли признаков - n предыдущих наблюдений ряда:
$$
\hat y _{t+1}(w) = \sum_{j=1}^n w_jy_{t-j+1}
$$
В роли объектов - l = t - n + 1 моментов в истории ряда.

**Автокорреляция** - с ее помощью можно квантифицировать степень сходства между значениями ряда в соседних точках (например в соседних месяцах):
$$
r_{\tau} = \frac{E((y_t - Ey)(y_{t+\tau} - Ey))}{Dy}
\\
по выборке:
\\
r_{\tau} = \frac{\sum_{t=1}^{T- \tau} (y_t - \bar y)(y_{t+\tau} - \bar y)}{\sum_{t=1}^T (y_t -\bar y)^2}
$$
\tau - лаг автокорреляции.

По сути автокорреляция - обычная корреляция Пирсона между исходным рядом и его версией, сдвинутой на несколько отсчетов (например месяцев), количество отсчетов называется лаг автокорреляции.

Отдельные временные ряды не имеют тенденции или циклической компоненты, каждый их следующий уровень равен сумме среднего уровня ряда и случайной компоненты. Такие ряды носят название **стационарные ряды.**

Проверить стационарен ли ряд можно с помощью критерия Дики-Фуллера и KPSS.

Если ряд не стационарен можно привести его к такому:

1. Стабилизация дисперсии. - Для монотонно меняющейся дисперсии можно использовать стабилизирующее преобразование (часто используют логарифмирование). 
2. Дифференцирование - переход к попарным разностям соседних значений. Это позволяет стабилизировать среднее значение ряда и избавиться от тренда. (Может применяться неоднократно).

**ARMA**

**Авторегрессия** - будем делать регрессию ряда не на какие то внешние признаки, зависящие от времени, а непосредственно на его собственные значения в прошлом:
$$
y_t = \alpha + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \epsilon_t
$$
Такая модель называется моделью авторегрессии порядка р - р(AR(p)): y_t - линейная комбинация р предыдущих значений ряда и шумовой компоненты. 

**Модели скользящего среднего** -  функция, значение который в каждой точке определения равны некоторому среднему значению исходной функции за предыдущий период. Обобщим модель:
$$
y_t = \alpha + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}
$$
Такая модель называется модель скользящего среднего порядка q(MA(q)): y_t - линейная комбинация q последних значений шумовой компоненты.

**ARMA(p, q)**:
$$
y_t = \alpha + \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}
$$
**Теорема Вольда**: Любой стационарный временной ряд может быть описан моделью ARMA(p, q).



**ARIMA(p, d, q)** -  модель ARMA(p, q) для d раз продифференцированного ряда.

https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

**SARMA(p, q) * (P,Q)** - модель ARMA + P авторегрессионных компонент + Q компонент скользящего окна. (Пусть ряд имеет сезонный период длины S)
$$
авторегрессионные: +\phi_Sy_{t-S} + \phi_{2S}y_{t-2S} + ... + \phi_{PS}y_{t-PS}
\\
скользящегосреднего: +\theta_S\epsilon_{t-S} + \theta_{2S}\epsilon_{t-2S} + ... + \theta_{PS}\epsilon_{t-PS}
$$
**SARIMA(p, d, q) * (P, D, Q)** - модель SARMA для ряда, к которому d раз было применено обычное дифференцирование и D раз - сезонное.



**Настройка параметров:**

1. \alpha, \phi, \theta - если все остальные параметры фиксированны, коэффициенты подбираются МНК. Чтобы найти коэффициенты \theta, шумовая компонента предварительно оценивается с помощью остатков авторегрессии.
2. d, D (порядки дифференцирования) - подбираются так, чтобы ряд стал стационарным. Если ряд сезонный, рекомендуется начинать с сезонного дифференцирования (D). Чем меньше раз мы продифференцируем, тем меньше будет дисперсия итогового прогноза.
3. q, Q, p, P - для сравнения моделей с разными гиперпараметрами можно использовать критерий Акаике. Начальное приближение можно выбрать с помощью автокорреляции.

- **p**: Trend autoregression order.
- **d**: Trend difference order.
- **q**: Trend moving average order. - максимальный значимый лаг
- **P**: Seasonal autoregressive order. - последний значимый сезонный лаг
- **D**: Seasonal difference order. 
- **Q**: Seasonal moving average order.
- **m**: The number of time steps for a single seasonal period.

**Подбор ARIMA:**

1. Смотрим на ряд
2. При необходимости стабилизируем дисперсию (преборазование Бокса-Кокса/логарифмирование)
3. Если ряд нестационарен, подбираем порядок дифференцирования
4. Анализируем графики автокорреляционной функции и частичной автокорреляционной функции, определяем примерные значения p, q, P, Q.
5. Обучаем модели-кандидаты, сравниваем их по критерию Акаике, выбираем победителя
6. Смотрим на остатки полученной модели, если они плохие пробуем что то поменять


















