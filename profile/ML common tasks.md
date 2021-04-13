[**Machine Learning techniques and tasks**](#Machine-Learning-techniques-and-tasks)

* [Learning to label sequence](#Learning-to-label-sequence)
* [Active learning](#Active learning)
* [Semi-supervised learning](#Semi-supervised learning)
* [One-shot learning](#One-shot learning)
* [Zero-shot learning](#Zero-shot learning)
* [Handling Multiple Inputs](#Handling-Multiple-Inputs)
* [Handling Multiple Outputs](#Handling-Multiple-Outputs)
* [Transfer Learning](#Transfer Learning)
* [Learning to Rank](#Learning to Rank)
* [Learning to Recommend](#Learning to Recommend)
* [Прогнозирование временных рядов](#Прогнозирование временных рядов)



NLP

* [Sentiment](#Sentiment)







## Machine Learning techniques and tasks



### Learning to label sequence

Sequence labeling is the problem of automatically assigning a label to each element of a sequence. A labeled sequential training example in sequence labeling is a pair of lists (X, Y), where X is a list of feature vectors, one per time step, Y is a list of the same length of labels. For example, X could represent words in a sentence such as [“big”, “beautiful”, “car”], and Y would be the list of the corresponding parts of speech, such as [“adjective”, “adjective”, “noun”]).

Models for label sequence:

1. RNN

2. Conditional Random Fields (CRF). 

   For example, imagine we have the task of named entity extraction and we want to build a model that would label each word in the sentence such as “I go to San Francisco” with one of the following classes: {location, name, company_name, other}. If our feature vectors (which represent words) contain such binary features as “whether or not the word starts with a capital letter” and “whether or not the word can be found in the list of locations,” such features would be very informative and help to classify the words San and Francisco as location



### Sequence-to-sequence learning 

Sequence-to-sequence learning (often abbreviated as seq2seq learning) is a generalization of the sequence labeling problem. In seq2seq, Xi and Yi can have different lengths. seq2seq models have found application in machine translation (where, for example, the input is an English sentence, and the output is the corresponding French sentence), conversational interfaces (where the input is a question typed by the user, and the output is the answer from the machine), text summarization, spelling correction, and many others.

Many but not all seq2seq learning problems are currently best solved by neural networks. The network architectures used in seq2seq all have two parts: an encoder and a decoder. 

In seq2seq neural network learning, the encoder is a neural network that accepts sequential input. It can be an RNN, but also a CNN or some other architecture. The role of the encoder is to read the input and generate some sort of state (similar to the state in RNN) that can be seen as a numerical representation of the meaning of the input the machine can work with. The meaning of some entity, whether it be an image, a text or a video, is usually a vector or a matrix that contains real numbers. This vector (or matrix) is called in the machine learning jargon the embedding of the input.

The decoder is another neural network that takes an embedding as input and is capable of generating a sequence of outputs. As you could have already guessed, that embedding comes from the encoder. To produce a sequence of outputs, the decoder takes a start of sequence input feature vector x(0) (typically all zeroes), produces the first output y(1), updates its state by combining the embedding and the input x(0), and then uses the output y(1) as its next input x(1). 

More accurate predictions can be obtained using an architecture with **attention**. Attention mechanism is implemented by an additional set of parameters that combine some information from the encoder (in RNNs, this information is the list of state vectors of the last recurrent layer from all encoder time steps) and the current state of the decoder to generate the label. That allows for even better retention of long-term dependencies than provided by gated units and bidirectional RNN.

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

### Handling Multiple Inputs

With neural networks, you have more flexibility. You can build two subnetworks, one for each type of input. For example, a CNN subnetwork would read the image while an RNN subnetwork would read the text. Both subnetworks have as their last layer an embedding: CNN has an embedding of the image, while RNN has an embedding of the text. You can now concatenate two embeddings and then add a classification layer, such as softmax or sigmoid, on top of the concatenated embeddings. Neural network libraries provide simple to use tools that allow concatenating or averaging layers from several subnetworks.

### Handling Multiple Outputs

In some cases the outputs are multimodal, and their combinations cannot be effectively enumerated. Consider the following example: you want to build a model that detects an object on an image and returns its coordinates. In addition, the model has to return a tag describing the object, such as “person,” “cat,” or “hamster.” Your training examples will a feature vector that represents an image. The label will be represented as a vector of coordinates of the object and another vector with a one-hot encoded tag. To handle a situation like that, you can create one subnetwork that would work as an encoder. It will read the input image using, for example, one or several convolution layers. The encoder’s last layer would be the embedding of the image. Then you add two other subnetworks on top of the embedding layer: one that takes the embedding vector as input and predicts the coordinates of an object. This first subnetwork can have a ReLU as the last layer, which is a good choice for predicting positive real numbers, such as coordinates; this subnetwork could use the mean squared error cost C1. The second subnetwork will take the same embedding vector as input and predict the probabilities for each label. This second subnetwork can have a softmax as the last layer, which is appropriate for the probabilistic output, and use the averaged negative log-likelihood cost C2 (also called cross-entropy cost).

### Transfer Learning

Transfer learning is probably where neural networks have a unique advantage over the shallow models. In transfer learning, you pick an existing model trained on some dataset, and you adapt this model to predict examples from another dataset, different from the one the model was built on. This second dataset is not like holdout sets you use for validation and test. It may represent some other phenomenon, or, as machine learning scientists say, it may come from another statistical distribution.

With neural networks, the situation is much more favorable. Transfer learning in neural networks works like this. 

1. You build a deep model on the original big dataset (wild animals). 
2. You compile a much smaller labeled dataset for your second model (domestic animals). 
3. You remove the last one or several layers from the first model. Usually, these are layers responsible for the classification or regression; they usually follow the embedding layer. 
4. You replace the removed layers with new layers adapted for your new problem. 
5. You “freeze” the parameters of the layers remaining from the first model. 
6. You use your smaller labeled dataset and gradient descent to train the parameters of only the new layers.

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









## Sentiment



Whether you use raw single-word tokens, n-grams, stems, or lemmas in your NLP pipeline, each of those tokens contains some information. An important part of this information is the word’s sentiment—the overall feeling or emotion that the word invokes. This sentiment analysis—measuring the sentiment of phrases or chunks of text—is a common application of NLP. In many companies it’s the main thing an NLP engineer is asked to do. Companies like to know what users think of their products. So they often will provide some way for you to give feedback. A star rating on Amazon or Rotten Tomatoes is one way to get quantitative data about how people feel about products they’ve purchased. But a more natural way is to use natural language comments.

There are two approaches to sentiment analysis: 

* A rule-based (heuristics) algorithm composed by a human. A common rule-based approach to sentiment analysis is to find keywords in the text and map each one to numerical scores or weights in a dictionary.
* A machine learning model learned from data by a machine. Relies on a labeled set of statements or documents to train a machine learning model to create those rules.



