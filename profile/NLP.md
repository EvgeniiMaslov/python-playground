## NLP Content



### Preprocessing

##### Basic preprocessing steps

* [Tokenizers](#Tokenizers)
* [Normalizing](#Normalizing)
* [Stop words](#Stop words)
* [Bag of words](#Bag of words)
* [n-Gram Language Models](#n-gram-language-models)
* [TFIDF](#tfidf)

##### Compute topic vectors that capture the semantics

* [Latent Dirichlet Allocation](#Latent Dirichlet Allocation)
  * [Latent Dirichlet Allocation + Linear Discriminant Analysis for sms](#Latent Dirichlet Allocation + Linear Discriminant Analysis for sms)
* [Linear Discriminant Analysis for semantic analysis](#Linear Discriminant Analysis for semantic analysis)
  * [LDA Example: The SMS spam classifier](#LDA Example: The SMS spam classifier)
* [Latent semantic analysis](#Latent semantic analysis)
* [Principal Component Analysis](#principal-component-analysis)
  * [PCA for SMS spam classifier](#PCA for SMS spam classifier)

##### Compute topic-word vectors that capture the semantics

* [Word Vectors](#Word Vectors)
* [Word2vec](#Word2vec)
  * [Train word2vec](#Train word2vec)
  * [Tricks for word2vec](#Tricks for word2vec)
  * [Word2vec Example](#Word2vec Example)
  * [Word2vec vs LSA](#Word2vec vs LSA)
  * [Document similarity with Doc2vec](#Document similarity with Doc2vec)
* [GloVe](#GloVe)
* [fastText](#fastText)



### ML models for NLP

##### Neural networks

* [CNN for NLP](#CNN for NLP)
* [RNN](#RNN)
  * [Common RNN challenges](#Common RNN challenges)
  * [Simple RNN model example](#Simple RNN model example)
* [Bidirectional RNN model example](#Bidirectional RNN model example)
* [LSTM](#LSTM)
* [GRU](#GRU)

##### Rule-based models

* [VADER - A rule-based sentiment analyzer](#VADER - A rule-based sentiment analyzer)



### NLP common task and solution



##### Building chatbot

* [Dialog engines](#Dialog engines)
* [Pattern-matching approach](#Pattern-matching approach)
* [Retrieval (search)](#Retrieval (search))
* [Generative models](# Generative models)
* [Pros and cons of each approach](#Pros and cons of each approach)



##### [Sentiment](#Sentiment)

[Sequence-to-sequence learning ](#Sequence-to-sequence learning )



##### Real-world NLP challenges

* [Information extraction (named entity extraction and question answering)](#Information extraction (named entity extraction and question answering))

  * [Named entities and relations](#Named entities and relations)
  * [Regular patterns](#Regular patterns)
  * [Information worth extracting](#Information worth extracting)
    * [Useful regular expression](#Useful regular expression)

  * [Extracting relationships (relations)](#Extracting relationships (relations))



## Preprocessing



### Tokenizers

You can use the NLTK function RegexpTokenizer to replicate your simple tokenizer example like this:

```python
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
tokenizer.tokenize(sentence)
```

An even better tokenizer is the Treebank Word Tokenizer from the NLTK package. It incorporates a variety of common rules for English word tokenization. For example, it separates phrase-terminating punctuation (?!.;,) from adjacent tokens and retains decimal numbers containing a period as a single token. In addition it contains rules for English contractions. For example “don’t” is tokenized as ["do", "n’t"] (it’s important to separate the words to allow the syntax tree parser to have a consistent, predictable set of tokens with known grammar rules as its input).

```python
from nltk.tokenize import TreebankWordTokenizer
sentence = """Monticello wasn't designated as UNESCO World Heritage Site until 1987."""
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize(sentence)
```

The NLTK library includes a tokenizer—casual_tokenize—that was built to deal with short, informal, emoticon-laced texts from social networks where grammar and spelling conventions vary widely.

```python
from nltk.tokenize.casual import casual_tokenize
message = """RT @TJMonticello Best day everrrrrrr at Monticello. Awesommmmmmeeeeeeee day :*)"""
casual_tokenize(message)
```



### Normalizing



**Case folding.** Words can become case “denormalized” when they are capitalized because of their presence at the beginning of a sentence, or when they’re written in ALL CAPS for emphasis. Undoing this denormalization is called case normalization, or more commonly, case folding. Normalizing word and character capitalization is one way to reduce your vocabulary size and generalize your NLP pipeline. However, some information is often communicated by capitalization of a word— for example, 'doctor' and 'Doctor' often have different meanings.

The simplest and most common way to normalize the case of a text string is to lowercase all the characters with a function like Python’s built-in str.lower(). Unfortunately this approach will also “normalize” away a lot of meaningful capitalization in addition to the less meaningful first-word-in-sentence capitalization you intended to normalize away. A better approach for case normalization is to lowercase only the first word of a sentence and allow all other words to retain their capitalization.

Even with this careful approach to case normalization, where you lowercase words only at the start of a sentence, you will still introduce capitalization errors for the rare proper nouns that start a sentence. “Joe Smith, the word smith, with a cup of joe.” will produce a different set of tokens than “Smith the word with a cup of joe, Joe Smith.”

To avoid this potential loss of information, many NLP pipelines don’t normalize for case at all. For many applications, the efficiency gain (in storage and processing) for reducing one’s vocabulary size by about half is outweighed by the loss of information for proper nouns. But some information may be “lost” even without case normalization. If you don’t identify the word “The” at the start of a sentence as a stop word, that can be a problem for some applications. Really sophisticated pipelines will detect proper nouns before selectively normalizing the case for words at the beginning of sentences that are clearly not proper nouns. You should implement whatever case normalization approach makes sense for your application.



**Stemming.** Another common vocabulary normalization technique is to eliminate the small meaning differences of pluralization or possessive endings of words, or even various verb forms. Stemming lead to big improvement in the “recall” score, but stemming could greatly reduce the “precision” score.

Two of the most popular stemming algorithms are the Porter and Snowball stemmers.

```python
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
' '.join([stemmer.stem(w).strip("'") for w in "dish washer's washed dishes".split()])
```



**Lemmatization.** If you have access to information about connections between the meanings of various words, you might be able to associate several words together even if their spelling is quite different. This more extensive normalization down to the semantic root of a word—its lemma—is called lemmatization.

Lemmatization is a potentially more accurate way to normalize a word than stemming or case normalization because it takes into account a word’s meaning. A lemmatizer uses a knowledge base of word synonyms and word endings to ensure that only words that mean similar things are consolidated into a single token.

Some lemmatizers use the word’s part of speech (POS) tag in addition to its spelling to help improve accuracy. The POS tag for a word indicates its role in the grammar of a phrase or sentence.

So lemmatizers are better than stemmers for most applications. Stemmers are only really used in large-scale information retrieval applications (keyword search). And if you really want the dimension reduction and recall improvement of a stemmer in your information retrieval pipeline, you should probably also use a lemmatizer right before the stemmer. Because the lemma of a word is a valid English word, stemmers work well on the output of a lemmatizer. This trick will reduce your dimensionality and increase your information retrieval recall even more than a stemmer alone.

```python
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("better")
```

**When should you use a lemmatizer or a stemmer? ** Stemmers are generally faster to compute and require less-complex code and datasets. But stemmers will make more errors and stem a far greater number of words, reducing the information content or meaning of your text much more than a lemmatizer would.

If your application involves search, stemming and lemmatization will improve the recall of your searches by associating more documents with the same query words. However, stemming, lemmatization, and even case folding will significantly reduce the precision and accuracy of your search results. These vocabulary compression approaches will cause an information retrieval system (search engine) to return many documents not relevant to the words’ original meanings. Because search results can be ranked according to relevance, search engines and document indexes often use stemming or lemmatization to increase the likelihood that the search results include the documents a user is looking for. But they combine search results for stemmed and unstemmed versions of words to rank the search results that they present to you.

For a search-based chatbot, however, accuracy is more important. As a result, a chatbot should first search for the closest match using unstemmed, unnormalized words before falling back to stemmed or filtered token matches to find matches. It should rank such matches of normalized tokens lower than the unnormalized token matches.







### Stop words

Stop words are common words in any language that occur with a high frequency but carry much less substantive information about the meaning of a phrase (for example: a, an, the, this, etc.).

Historically, stop words have been excluded from NLP pipelines in order to reduce the computational effort to extract information from a text. Even though the words themselves carry little information, the **stop words can provide important relational information** as part of an n-gram. 

Consider these two examples: 

* Mark reported to the CEO 
* Suzanne reported as the CEO to the board 

In your NLP pipeline, you might create 4-grams such as reported to the CEO and reported as the CEO. If you remove the stop words from the 4-grams, both examples would be reduced to "reported CEO", and you would lack the information about the professional hierarchy. 

Unfortunately, **retaining the** **stop words** within your pipeline **creates another problem**: it increases the length of the n-grams required to make use of these connections formed by the otherwise meaningless stop words. 

So if you have sufficient memory and processing bandwidth to run all the NLP steps in your pipeline on the larger vocabulary, you probably don’t want to worry about ignoring a few unimportant words here and there. And if you’re worried about overfitting a small training set with a large vocabulary, there are better ways to select your vocabulary or reduce your dimensionality than ignoring stop words.









### Bag of words

Bag of words is also called a word frequency vector, because it only counts the frequency of words, not their order. You could use this single vector to represent the whole document or sentence in a single, reasonable length vector.

**Binary.** Alternatively, if you’re doing basic keyword search, you could OR the one-hot word vectors into a binary bag-of-words vector. Search indexes only need to know the presence or absence of each word in each document to help you find those documents later.

```python
# build vocabulary
wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
            
# select most frequently occurring words
import heapq
most_freq = heapq.nlargest(2000, wordfreq, key=wordfreq.get)


# convert the sentences in our corpus into their corresponding vector representation
sentence_vectors = []
for sentence in corpus:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)
```



**Measuring bag-of-words overlap**. If we can measure the bag of words overlap for two vectors, we can get a good estimate of how similar they are in the words they use. And this is a good estimate of how similar they are in meaning. You can use dot product to estimate the bag-of-words vector overlap between some new sentences and the original sentence.

```python
>>> np.dot(sentence0, sentence1)
1 # one word was used in both sentences
```





### n-Gram Language Models

An n-gram is a sequence containing up to n elements that have been extracted from a sequence of those elements, usually a string. In general the “elements” of an n-gram can be characters, syllables, words, or even symbols like “A,” “T,” “G,” and “C” used to represent a DNA sequence. When a sequence of tokens is vectorized into a bag-of-words vector, it loses a lot of the meaning inherent in the order of those words. By extending your concept of a token to include multiword tokens, n-grams, your NLP pipeline can retain much of the meaning inherent in the order of words in your statements.

If tokens or n-grams are extremely rare, they don’t carry any correlation with other words that you can use to help identify topics or themes that connect documents or classes of documents. So rare n-grams won’t be helpful for classification problems.

```python
from nltk.util import ngrams
list(ngrams(tokens, 2))
```



```python
from collections import defaultdict
transitions = defaultdict(list)
for prev, current in zip(document, document[1:]):
 transitions[prev].append(current)

def generate_using_bigrams() -> str:
 current = "." # this means the next word will start a sentence
 result = []
 while True:
 	next_word_candidates = transitions[current] # bigrams (current, _)
 	current = random.choice(next_word_candidates) # choose one at random
 	result.append(current) # append it to results
 	if current == ".": return " ".join(result) # if "." we're done
```

```python
trigram_transitions = defaultdict(list)
starts = []
for prev, current, next in zip(document, document[1:], document[2:]):
 if prev == ".": # if the previous "word" was a period
 	starts.append(current) # then this is a start word
 trigram_transitions[(prev, current)].append(next)

def generate_using_trigrams() -> str:
 current = random.choice(starts) # choose a random starting word
 prev = "." # and precede it with a '.'
 result = [current]
 while True:
 	next_word_candidates = trigram_transitions[(prev, current)]
 	next_word = random.choice(next_word_candidates)
 	prev, current = current, next_word
 	result.append(current)
 	if current == ".":
 		return " ".join(result)
```

### TFIDF

**Zipf’s law** states that given some corpus of natural language utterances, the frequency of any word is inversely proportional to its rank in the frequency table.

A good way to think of a term’s inverse document frequency is this: How strange is it that this token is in this document? If a term appears in one document a lot of times, but occurs rarely in the rest of the corpus, one could assume it’s important to that document specifically.

When we are analyzing text data, we often encounter words that occur across multiple documents from both classes. These frequently occurring words typically don't contain useful or discriminatory information. In this subsection, you will learn about a useful technique called the term frequency-inverse document frequency (tf-idf), which can be used to downweight these frequently occurring words in the feature vectors. The tf-idf can be defined as the product of the term frequency and the inverse document frequency (for a given term, t, in a given document, d, in a corpus): 
$$
𝑡f𝑖df(𝑡, 𝑑) = 𝑡f(𝑡, 𝑑) × 𝑖df(𝑡, 𝑑) 
\\
or
\\
𝑡f𝑖df(𝑡, 𝑑) = 𝑡f(𝑡, 𝑑) × (1 + 𝑖df(𝑡, 𝑑))
$$

tf(t, d)—the number of times a term, t, occurs in a document, d. It should be noted that, in the bag-of-words model, the word or term order in a sentence or document does not matter. The order in which the term frequencies appear in the feature vector is derived from the vocabulary indices, which are usually assigned alphabetically.
$$
tf(t, d) = \frac{count(t)}{count(d)}
\\
idf(t, d) = log(\frac{n_d}{1 + df(d, t)}) = log\frac{number_.of_.documents}{number_.of_.documents_.containing_.t}
\\
or
\\
idf(t, d) = log(\frac{1 + n_d}{1 + df(d, t)})
$$
Here, nd is the total number of documents, and df(d, t) is the number of documents, d, that contain the term t. 



To make sure that we understand how TfidfTransformer works, let's walk through an example and calculate the tf-idf of the word 'is' in the third document. The word 'is' has a term frequency of 3 (tf = 3) in the third document, and the document frequency of this term is 3 since the term 'is' occurs in all three documents (df = 3). Thus, we can calculate the inverse document frequency as follows:
$$
idf('is', d_3) = log\frac{1 + 3}{1 + 3} = 0
\\
tfidf('is', 3) = 3 × (0 + 1) = 3
$$
If we repeated this calculation for all terms in the third document, we'd obtain the following tf-idf vectors: [3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0, 1.69, 1.29]. However, notice that the values in this feature vector are different from the values that we obtained from TfidfTransformer that we used previously. The final step that we are missing in this tf-idf calculation is the L2-normalization, which can be applied as follows:
$$
tfidf(d_3)_{norm} = \frac{[3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0, 1.69, 1.29]}{\sqrt{3.39^2 + 3^2 + 3.39^2 + 1.29^2 + 1.29^2 + 1.29^2 + 2^2 + 1.69^2 + 1.29^2}}=
\\
= [0.5, 0.45, 0.5, 0.19, 0.19, 0.19, 0.3, 0.25, 0.19]
$$





Alternative TF-IDF normalization approaches:

![](C:\Users\sqrte\python-playground\profile\img\tfidf_alt.png)













## Compute topic vectors that capture the semantics



### Latent Dirichlet Allocation

LDA is a generative probabilistic model that tries to find groups of words that appear frequently together across different documents. These frequently appearing words represent our topics, assuming that each document is a mixture of different words. The input to an LDA is the bag-of-words model that we discussed earlier in this chapter. Given a bag-of-words matrix as input, LDA decomposes it into two new matrices: 

* A document-to-topic matrix 
* A word-to-topic matrix

LDA decomposes the bag-of-words matrix in such a way that if we multiply those two matrices together, we will be able to reproduce the input, the bag-of-words matrix, with the lowest possible error. 



In particular, we have a collection of documents, each of which is a list of words. And we have a corresponding collection of document_topics that assigns a topic (here a number between 0 and K – 1) to each word in each document. We can estimate the likelihood that topic 1 produces a certain word by comparing how many times topic 1 produces that word with how many times topic 1 produces any word. 

We start by assigning every word in every document a topic completely at random. Now we go through each document one word at a time. For that word and document, we construct weights for each topic that depend on the (current) distribution of topics in that document and the (current) distribution of words for that topic. We then use those weights to sample a new topic for that word. If we iterate this process many times, we will end up with a joint sample from the topic–word distribution and the document– topic distribution.





The LDA approach was developed in 2000 by geneticists in the UK to help them “infer population structure” from sequences of genes. Stanford Researchers (including Andrew Ng) popularized the approach for NLP in 2003. 

They imagined a machine that only had two choices to make to get started generating the mix of words for a particular document. They imagined that the document generator chose those words randomly, with some probability distribution over the possible choices. Your document “character sheet” needs only two rolls of the dice.  The two rolls of the dice represent the:

1. Number of words to generate for the document (Poisson distribution) 
2. Number of topics to mix together for the document (Dirichlet distribution)

After it has these two numbers, the hard part begins, choosing the words for a document. The imaginary BOW generating machine iterates over those topics and randomly chooses words appropriate to that topic until it hits the number of words that it had decided the document should contain in step 1.  Deciding the probabilities of those words for topics—the appropriateness of words for each topic—is the hard part. But once that has been determined, your “bot” just looks up the probabilities for the words for each topic from a matrix of term-topic probabilities. 

So all this machine needs is a single parameter for that Poisson distribution (in the dice roll from step 1) that tells it what the “average” document length should be, and a couple more parameters to define that Dirichlet distribution that sets up the number of topics. Then your document generation algorithm needs a term-topic matrix of all the words and topics it likes to use, its vocabulary. And it needs a mix of topics that it likes to “talk” about.

**Estimating topics using LDA.** Blei and Ng realized that they could determine the parameters for steps 1 and 2 by analyzing the statistics of the documents in a corpus. For example, for step 1, they could calculate the mean number of words (or n-grams) in all the bags of words for the documents in their corpus;  Keep in mind, you should calculate this statistic directly from your BOWs. You need to make sure you’re counting the tokenized and vectorized (Counter()-ed) words in your documents. And make sure you’ve applied any stop word filtering, or other normalizations before you count up your unique terms.

The second parameter you need to specify for an LDA model, the number of topics, is a bit trickier. The number of topics in a particular set of documents can’t be measured directly until after you’ve assigned words to those topics. Once you’ve told LDA how many topics to look for, it will find the mix of words to put in each topic to optimize its objective function.



##### Latent Dirichlet Allocation + Linear Discriminant Analysis for sms



The topics produced by LDA tend to be more understandable and “explainable” to humans. This is because words that frequently occur together are assigned the same topics, and humans expect that to be the case.

```python
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
np.random.seed(42)
counter = CountVectorizer(tokenizer=casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(), index=index)

column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),counter.vocabulary_.keys())))
bow_docs.columns = terms


from sklearn.decomposition import LatentDirichletAllocation as LDiA
ldia = LDiA(n_components=16, learning_method='batch')
ldia = ldia.fit(bow_docs)

ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,index=index, columns=columns)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors, sms.spam, test_size=0.5, random_state=271828)

lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors)
```



### Linear Discriminant Analysis for semantic analysis

LDA is a method that tries to maximize the distance of points belonging to different classes while minimizing the distance of points of the same class.

The general concept behind LDA is very similar to PCA, but whereas PCA attempts to find the orthogonal component axes of maximum variance in a dataset, the goal in LDA is to find the feature subspace that optimizes class separability.

All you need to “train” an LDA model is to find the vector (line) between the two centroids for your binary class. LDA is a supervised algorithm, so you need labels for your messages. To do inference or prediction with that model, you just need to find out if a new TF-IDF vector is closer to the in-class (spam) centroid than it is to the out-of-class (nonspam) centroid. First let’s “train” an LDA model to classify SMS messages as spam or nonspam (see the following listing). 

##### LDA Example: The SMS spam classifier

```python
# Get data
import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120
sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
sms['spam'] = sms.spam.astype(int)

# Tokenization
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()

# compute the centroids of your binary class
mask = sms.spam.astype(bool).values
spam_centroid = tfidf_docs[mask].mean(axis=0)
ham_centroid = tfidf_docs[~mask].mean(axis=0)

spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)

# spamminess_score is the distance along the line from the ham centroid to the spam centroid

# get predictions
from sklearn.preprocessing import MinMaxScaler
sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
sms['lda_predict'] = (sms.lda_score > .5).astype(int)
```







### Latent semantic analysis

Latent semantic analysis is based on the oldest and most commonly-used technique for dimension reduction, singular value decomposition. One application of SVD is matrix inversion. A matrix can be inverted by decomposing it into three simpler square matrices, transposing matrices, and then multiplying them back together.

Using SVD, LSA can break down your TF-IDF term-document matrix into three simpler matrices. And they can be multiplied back together to produce the original matrix, without any changes. This is like factorization of a large integer. But these three simpler matrices from SVD reveal properties about the original TFIDF matrix that you can exploit to simplify it.

When you use SVD this way in natural language processing, you call it latent semantic analysis. LSA uncovers the semantics, or meaning, of words that is hidden and waiting to be uncovered.

Latent semantic analysis is a mathematical technique for finding the “best” way to linearly transform (rotate and stretch) any set of NLP vectors, like your TF-IDF vectors or bag-of-words vectors. And the “best” way for many applications is to line up the axes (dimensions) in your new vectors with the greatest “spread” or variance in the word frequencies. You can then eliminate those dimensions in the new vector space that don’t contribute much to the variance in the vectors from document to document. Using SVD this way is called truncated singular value decomposition (truncated SVD).

LSA on natural language documents is equivalent to PCA on TF-IDF vectors.

LSA tells you which dimensions in your vector are important to the semantics (meaning) of your documents. You can discard those dimensions (topics) that have the least amount of variance between documents. These low-variance topics are usually distractions, noise, for any machine learning algorithm.



### Principal component analysis

LSA on natural language documents is equivalent to PCA on TF-IDF vectors.

PCA is dimensionality reduction algorithm. Dimension reduction is the primary countermeasure for overfitting. By consolidating your dimensions (words) into a smaller number of dimensions (topics), your NLP pipeline will become more “general.” Your spam filter will work on a wider range of SMS messages if you reduce your dimensions, or “vocabulary.”

##### PCA for SMS spam classifier

```python
# load data
import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120
sms = get_data('sms-spam')

index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]
sms.index = index

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()

tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean()

# pca
from sklearn.decomposition import PCA
pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)

'''
	If you’re curious about these topics, you can find out how much of each word they
	“contain” by examining their weights.
'''

column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(),tfidf.vocabulary_.keys())))
weights = pd.DataFrame(pca.components_, columns=terms, index=['topic{}'.format(i) for i in range(16)])
pd.options.display.max_columns = 8
weights.head(4).round(3)
```



### Word Vectors

Word vectors are numerical vector representations of word semantics, or meaning, including literal and implied meaning.

One important innovation involves representing words as low-dimensional vectors. These vectors can be compared, added together, fed into machine learning models, or anything else you want to do with them. 

Coming up with such vectors for a large vocabulary of words is a difficult undertaking, so typically we’ll learn them from a corpus of text. There are a couple of different schemes, but at a high level the task typically looks something like this:

1. Get a bunch of text. 
2. Create a dataset where the goal is to predict a word given nearby words (or alternatively, to predict nearby words given a word). 
3. Train a neural net to do well on this task. 
4. Take the internal states of the trained neural net as the word vectors.

With word vectors, questions like Portland Timbers + Seattle - Portland = ? can be solved with vector algebra.

Word vectors are biased! Word vectors learn word relationships based on the training corpus. If your corpus is about finance then your “bank” word vector will be mainly about businesses that hold deposits. If your corpus is about geology, then your “bank” word vector will be trained on associations with rivers and streams. And if you corpus is mostly about a matriarchal society with women bankers and men washing clothes in the river, then your word vectors would take on that gender bias.



### Word2vec

Topic vectors constructed from entire documents using LSA are great for document classification, semantic search, and clustering. But the topic-word vectors that LSA produces aren’t accurate enough to be used for semantic reasoning or classification and clustering of short phrases or compound words.

Word2vec learns the meaning of words merely by processing a large corpus of unlabeled text. The Word2vec model contains information about the relationships between words, including similarity. The Word2vec model “knows” that the terms Portland and Portland Timbers are roughly the same distance apart as Seattle and Seattle Sounders. And those distances (differences between the pairs of vectors) are in roughly the same direction. 

What you do care about is the internal representation, the vector that Word2vec gradually builds up to help it generate those predictions. This representation will capture much more of the meaning of the target word (its semantics) than the word-topic vectors that came out of latent semantic analysis and latent Dirichlet allocation.

#### Train word2vec

There are two possible ways to train Word2vec embeddings: 

* The skip-gram approach predicts the context of words (output words) from a word of interest (the input word). 
* The continuous bag-of-words (CBOW) approach predicts the target word (the output word) from the nearby words (input words). We show you how and when to use each of these to train a Word2vec model in the coming sections.

In the **skip-gram training approach**, you’re trying to predict the surrounding window of words based on an input word. If you want to train a Word2vec model using a skip-gram window size (radius) of two words, you’re considering the two words before and after each target word. You would then use your 5-gram tokenizer  to turn a sentence like this 

> sentence = "Claude Monet painted the Grand Canal of Venice in 1806." 

 into 10 5-grams with the input word at the center, one for each of the 10 words in the original sentence.

The training set consisting of the input word and the surrounding (output) words are now the basis for the training of the neural network. In the case of four surrounding words, you would use four training iterations, where each output word is being predicted based on the input word. Each of the words are represented as one-hot vectors before they are presented to the network.

The output vector for a neural network doing embedding is similar to a one-hot vector as well. The softmax activation of the output layer nodes (one for each token in the vocabulary) calculates the probability of an output word being found as a surrounding word of the input word. The output vector of word probabilities can then be converted into a one-hot vector where the word with the highest probability will be converted to 1, and all remaining terms will be set to 0. This simplifies the loss calculation.

After training of the neural network is completed, you’ll notice that the weights have been trained to represent the semantic meaning. Thanks to the one-hot vector conversion of your tokens, each row in the weight matrix represents each word from the vocabulary for your corpus. After the training, semantically similar words will have similar vectors, because they were trained to predict similar surrounding words. This is purely magical! After the training is complete and you decide not to train your word model any further, the output layer of the network can be ignored.

In the **continuous bag-of-words approach**, you’re trying to predict the center word based on the surrounding words. Instead of creating pairs of input and output tokens, you’ll create a multi-hot vector of all surrounding terms as an input vector. The multi-hot input vector is the sum of all one-hot vectors of the surrounding tokens to the center, target token.

**SKIP-GRAM VS. CBOW: WHEN TO USE WHICH APPROACH.** Mikolov highlighted that the skip-gram approach works well with small corpora and rare terms. With the skip-gram approach, you’ll have more examples due to the network structure. But the continuous bag-of-words approach shows higher accuracies for frequent words and is much faster to train. 

#### Tricks for word2vec

**Frequent bigrams.** Some words often occur in combination with other words. In order to improve the accuracy of the Word2vec embedding, Mikolov’s team included some bigrams and trigrams as terms in the Word2vec vocabulary. The team used co-occurrence frequency to identify bigrams and trigrams that should be considered single terms, using the following scoring function:
$$
score(w_i, w_j) = \frac{count(w_i, w_j) - \delta}{count(w_i) * count(w_j)}
$$
If the words wi and wj result in a high score and the score is above the threshold \delta, they will be included in the Word2vec vocabulary as a pair term.

Another effect of the word pairs is that the word combination often represents a different meaning than the individual words. For example, the MLS soccer team Portland Timbers has a different meaning than the individual words Portland and Timbers. But by adding oft-occurring bigrams like team names to the Word2vec model, they can easily be included in the one-hot vector for model training. 

**Subsampling frequent tokens.** Another accuracy improvement to the original algorithm was to subsample frequent words. Common words like “the” or “a” often don’t carry significant information. And the co-occurrence of the word “the” with a broad variety of other nouns in the corpus might create less meaningful connections between words, muddying the Word2vec representation with this false semantic similarity training.

To reduce the emphasis on frequent words like stop words, words are sampled during training in inverse proportion to their frequency. The effect of this is similar to the IDF effect on TF-IDF vectors. Frequent words are given less influence over the vector than the rarer words. Tomas Mikolov used the following equation to determine the probability of sampling a given word. This probability determines whether or not a particular word is included in a particular skip-gram during training:
$$
P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}
$$
In the preceding equations, f(w_i) represents the frequency of a word across the corpus, and t represents a frequency threshold above which you want to apply the subsampling probability.

**Negative sampling.** If a single training example with a pair of words is presented to the network, it’ll cause all weights for the network to be updated. This changes the values of all the vectors for all the words in your vocabulary. But if your vocabulary contains thousands or millions of words, updating all the weights for the large one-hot vector is inefficient. To speed up the training of word vector models, Mikolov used negative sampling.

Instead of updating all word weights that weren’t included in the word window, Mikolov suggested sampling just a few negative samples (in the output vector) to update their weights. Instead of updating all weights, you pick n negative example word pairs (words that don’t match your target output for that example) and update the weights that contributed to their specific output. That way, the computation can be reduced dramatically and the performance of the trained network doesn’t decrease significantly.

#### Word2vec Example

```python
from gensim.models.keyedvectors import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('/path/to/GoogleNews-vectors-negative300.bin.gz',\
                                                 binary=True, limit=200000)

word_vectors.most_similar(positive=['cooking', 'potatoes'], topn=5)
''' [('cook', 0.6973530650138855),
    ('oven_roasting', 0.6754530668258667),
    ('Slow_cooker', 0.6742032170295715),
    ('sweet_potatoes', 0.6600279808044434),
    ('stir_fry_vegetables', 0.6548759341239929)]
'''
# king - man + woman = queen
word_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=2)
# [('queen', 0.7118192315101624), ('monarch', 0.6189674139022827)]
```



#### Word2vec vs LSA

Advantages of LSA are 

* Faster training 
* Better discrimination between longer documents 

Advantages of Word2vec and GloVe are 

* More efficient use of large corpora 
* More accurate reasoning with words, such as answering analogy questions



#### Document similarity with Doc2vec

The concept of Word2vec can also be extended to sentences, paragraphs, or entire documents. The idea of predicting the next word based on the previous words can be extended by training a paragraph or document vector. In this case, the prediction not only considers the previous words, but also the vector representing the paragraph or the document. It can be considered as an additional word input to the prediction. Over time, the algorithm learns a document or paragraph representation from the training set.

**How to train document vectors.** 

```python
import multiprocessing
num_cores = multiprocessing.cpu_count()

'''
	gensim provides a data structure to
    annotate documents with string or
    integer tags for category labels,
    keywords, or whatever information you
    want to associate with your documents (TaggedDocument). 
'''
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess

corpus = ['This is the first document ...', 'another document ...']
training_corpus = []

for i, text in enumerate(corpus):
    tagged_doc = TaggedDocument(simple_preprocess(text), [i])
    
    training_corpus.append(tagged_doc)
    
model = Doc2Vec(size=100, min_count=2,
                workers=num_cores, iter=10)
model.build_vocab(training_corpus)
model.train(training_corpus, total_examples=model.corpus_count, epochs=model.iter)


# infer
model.infer_vector(simple_preprocess('This is a completely unseen document'), steps=10)
```

With these few steps, you can quickly train an entire corpus of documents and find similar documents. You could do that by generating a vector for every document in your corpus and then calculating the cosine distance between each document vector. Another common task is to cluster the document vectors of a corpus with something like k-means to create a document classifier. 



### GloVe



Word2vec was a breakthrough, but it relies on a neural network model that must be trained using backpropagation. Backpropagation is usually less efficient than direct optimization of a cost function using gradient descent. Stanford NLP researchers led by Jeffrey Pennington set about to understand the reason why Word2vec worked so well and to find the cost function that was being optimized. They started by counting the word co-occurrences and recording them in a square matrix. They found they could compute the singular value decomposition of this co-occurrence matrix, splitting it into the same two weight matrices that Word2vec produces. The key was to normalize the co-occurrence matrix the same way. But in some cases the Word2vec model failed to converge to the same global optimum that the Stanford researchers were able to achieve with their SVD approach. It’s this direct optimization of the global vectors of word co-occurrences (co-occurrences across the entire corpus) that gives GloVe its name.

GloVe can produce matrices equivalent to the input weight matrix and output weight matrix of Word2vec, producing a language model with the same accuracy as Word2vec but in much less time. GloVe speeds the process by using the text data more efficiently. GloVe can be trained on smaller corpora and still converge. And SVD algorithms have been refined for decades, so GloVe has a head start on debugging and algorithm optimization. Word2vec relies on backpropagation to update the weights that form the word embeddings. Neural network backpropagation is less efficient than more mature optimization algorithms such as those used within SVD for GloVe.

Even though Word2vec first popularized the concept of semantic reasoning with word vectors, your workhorse should probably be GloVe to train new word vector models. With GloVe you’ll be more likely to find the global optimum for those vector representations, giving you more accurate results. 

Advantages of GloVe are:

* Faster training 
* Better RAM/CPU efficiency (can handle larger documents) 
* More efficient use of data (helps with smaller corpora) 
* More accurate for the same amount of training 



### fastText



Researchers from Facebook took the concept of Word2vec one step further by adding a new twist to the model training. The new algorithm, which they named fastText, predicts the surrounding n-character grams rather than just the surrounding words, like Word2vec does. For example, the word “whisper” would generate the following 2- and 3-character grams: wh, whi, hi, his, is, isp, sp, spe, pe, per, er.

fastText trains a vector representation for every n-character gram, which includes words, misspelled words, partial words, and even single characters. The advantage of this approach is that it handles rare words much better than the original Word2vec approach.

**How to use pretrained fastText models.** The use of fastText is just like using Google’s Word2vec model. Head over to the fastText model repository and download the bin+text model for your language of choice. After the download finishes, unzip the binary language file.

```python
from gensim.models.fasttext import FastText
ft_model = FastText.load_fasttext_format(model_file=MODEL_PATH)
ft_model.most_similar('soccer')
```



### CNN for NLP



It turns out you can use convolutional neural networks for natural language processing by using word vectors (also known as word embeddings) or one-hot encoded vectors.

The weight values in the filters are unchanged for a given input sample during the forward pass, which means you can take a given filter and all its “snapshots” in parallel and compose the output “image” all at once. This is the convolutional neural network’s secret to speed. This speed, plus its ability to ignore the position of a feature, is why researchers keep coming back to this convolutional approach to feature extraction.

Why would you choose a CNN for your NLP classification task? The main benefit it provides is efficiency. In many ways, because of the pooling layers and the limits created by filter size (though you can make your filters large if you wish), you’re throwing away a good deal of information. But that doesn’t mean they aren’t useful models. As you’ve seen, they were able to efficiently detect and predict sentiment over a relatively large dataset, and even though you relied on the Word2vec embeddings, CNNs can perform on much less rich embeddings without mapping the entire language.





### RNN

RNN provide a way to remember what just happened the moment before (specifically what happened at time step t when you’re looking at time step t+1). Recurrent neural nets (RNNs) enable neural networks to remember the past words within a sentence.

Although the idea of affecting state across time can be a little mind boggling at first, the basic concept is simple. For each input you feed into a regular feedforward net, you’d like to take the output of the network at time step t and provide it as an additional input, along with the next piece of data being fed into the network at time step t+1. You tell the feedforward network what happened before along with what is happening “now.”



##### Common RNN challenges



* Vanishing and exploding gradients: As the lengths of these sequences increase, the gradients going back will become smaller and smaller. This will cause the network to train slowly or not learn at all. This effect will be more pronounced as sequence lengths increase. For managing exploding gradients, a technique called gradient clipping is used. This technique artificially clips gradients if their magnitude exceeds a threshold. This prevents gradients from becoming too large or exploding.
* Inability to manage long-term dependencies
* Two specific RNN cell designs mitigate these problems: Long-Short Term Memory (LSTM) and Gated Recurrent Unit (GRU).





##### Simple RNN model example

Notice here the keyword argument return_sequences. It’s going to tell the network to return the network value at each time step, hence the 400 vectors, each 50 long. If return_sequences was set to False (the Keras default behavior), only a single 50-dimensional vector would be returned.

Keras provides a keyword argument in the base RNN layer (therefore in the SimpleRNN as well) called stateful. It defaults to False. If you flip this to True when adding the SimpleRNN layer to your model, the last sample’s last output passes into itself at the next time step along with the first token input, just as it would in the middle of the sample.

```python
model = Sequential()
model.add(SimpleRNN(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))

model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
```

##### Bidirectional RNN model example



```python
model = Sequential()
model.add(Bidirectional(SimpleRNN(num_neurons, return_sequences=True),
                        input_shape=(maxlen, embedding_dims)))
```



### LSTM



Your challenge is to build a network that can pick up on the same core thought in both sentences. What you need is a way to remember the past across the entire input sequence. A long short-term memory (LSTM) is just what you need. Modern versions of a long short-term memory network typically use a special neural network unit called a gated recurrent unit (GRU). A gated recurrent unit can maintain both long- and short-term memory efficiently, enabling an LSTM to process a long sentence or document more accurately. In fact, LSTMs work so well they have replaced recurrent neural networks in almost all applications involving time series, discrete sequences, and NLP.

LSTM has four main parts: 

* Cell value or memory of the network, also referred to as the cell, which stores accumulated knowledge 
* Input gate, which controls how much of the input is used in computing the new cell value 
* Output gate, which determines how much of the cell value is used in the output 
* Forget gate, which determines how much of the current cell value is used for updating the cell value

LSTMs introduce the concept of a state for each layer in the recurrent network. The state acts as its memory. You can think of it as adding attributes to a class in object oriented programming. The memory state’s attributes are updated with each training example.

In LSTMs, the rules that govern the information stored in the state (memory) are trained neural nets themselves—therein lies the magic. They can be trained to learn what to remember, while at the same time the rest of the recurrent net learns to predict the target label!

With LSTMs, patterns that humans take for granted and process on a subconscious level begin to be available to your model.



### GRU



Compared to the LSTM, it has fewer gates. Input and forget gates are combined into a single update gate. Some of the internal cell state and hidden state is merged together as well. This reduction in complexity makes it easier to train. It has shown great results in the speech and sound domains. However, in neural machine translation tasks, LSTMs have shown superior performance.



## Sentiment analysis



#### VADER - A rule-based sentiment analyzer



Hutto and Gilbert at GA Tech came up with one of the first successful rule-based sentiment analysis algorithms. They called their algorithm VADER, for Valence Aware Dictionary for sEntiment Reasoning. Many NLP packages implement some form of this algorithm. The NLTK package has an implementation of the VADER algorithm in nltk.sentiment.vader.

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()

corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
          "Horrible! Completely useless. :(",
          "It was OK. Some good and some bad things."]
for doc in corpus:
    scores = sa.polarity_scores(doc)
    print('{:+}: {}'.format(scores['compound'], doc))
```

The only drawback is that VADER doesn’t look at all the words in a document, only about 7,500.





## Building chatbot



### Dialog engines

Chatbots have come a long way since the days of ELIZA. Pattern-matching technology has been generalized and refined over the decades. And completely new approaches have been developed to supplement pattern matching. In recent literature, chatbots are often referred to as dialog systems, perhaps because of this greater sophistication. Matching patterns in text and populating canned-response templates with information extracted with those patterns is only one of four modern approaches to building chatbots:

* Pattern matching—Pattern matching and response templates (canned responses) 
* Grounding—Logical knowledge graphs and inference on those graphs 
* Search—Text retrieval 
* Generative—Statistics and machine learning

This is roughly the order in which these approaches were developed. The most advanced chatbots use a hybrid approach that combines all of these techniques.  This hybrid approach enables them to accomplish a broad range of tasks.

**QUESTION ANSWERING DIALOG SYSTEMS** 

Question answering chatbots are used to answer factual questions about the world, which can include questions about the chatbot itself. Many question answering systems first search a knowledge base or relational database to “ground” them in the real world. If they can’t find an acceptable answer there, they may search a corpus of unstructured data (or even the entire Web) to find answers to your questions. This is essentially what Google Search does. Parsing a statement to discern the question in need of answering and then picking the right answer requires a complex pipeline that combines most of the elements covered in previous chapters. Question answering chatbots are the most difficult to implement well because they require coordinating so many different elements. 

**VIRTUAL ASSISTANTS** 

Virtual assistants, such as Alexa and Google Assistant, are helpful when you have a goal in mind. Goals or intents are usually simple things such as launching an app, setting a reminder, playing some music, or turning on the lights in your home. For this reason, virtual assistants are often called goal-based dialog engines. Dialog with such chatbots is intended to conclude quickly, with the user being satisfied that a particular action has been accomplished or some bit of information has been retrieved.

**CONVERSATIONAL CHATBOTS**

Conversational chatbots, such as Worswick’s Mitsuku or any of the Pandorabots,15 are designed to entertain. They can often be implemented with very few lines of code, as long as you have lots of data. But doing conversation well is an ever-evolving challenge. The accuracy or performance of a conversational chatbot is usually measured with something like a Turing test. In a typical Turing test, humans interact with another chat participant through a terminal and try to figure out if it’s a bot or a human. The better the chatbot is at being indistinguishable from a human, the better its performance on a Turing test metric.

**MARKETING CHATBOTS** 

Marketing chatbots are designed to inform users about a product and entice them to purchase it. More and more video games, movies, and TV shows are launched with chatbots on websites promoting them

**COMMUNITY MANAGEMENT** 

Community management is a particularly important application of chatbots because it influences how society evolves. A good chatbot “shepherd” can steer a video game community away from chaos and help it grow into an inclusive, cooperative world where everyone has fun, not just the bullies and trolls. A bad chatbot, such as the Twitter bot Tay, can quickly create an environment of prejudice and ignorance.

**CUSTOMER SERVICE** 

Customer service chatbots are often the only “person” available when you visit an online store. IBM’s Watson, Amazon’s Lex, and other chatbot services are often used behind the scenes to power these customer assistants. They often combine both question answering skills (remember Watson’s Jeopardy training?) with virtual assistance skills. But unlike marketing bots, customer service chatbots must be well-grounded. And the knowledge base used to “ground” their answers to reality must be kept current, enabling customer service chatbots to answer questions about orders or products as well as initiate actions such as placing or canceling orders.

**THERAPY** 

Modern therapy chatbots, such as Wysa and YourDOST, have been built to help displaced tech workers adjust to their new lives. Therapy chatbots must be entertaining like a conversational chatbot. They must be informative like a question answering chatbot. And they must be persuasive like a marketing chatbot. And if they’re imbued with self-interest to augment their altruism, these chatbots may be “goal seeking” and use their marketing and influence skill to get you to come back for additional sessions.



### Pattern-matching approach

The earliest chatbots used pattern matching to trigger responses. In addition to detecting statements that your bot can respond to, patterns can also be used to extract information from the incoming text.

The information extracted from your users’ statements can be used to populate a database of knowledge about the users, or about the world in general. And it can be used even more directly to populate an immediate response to some statements.

### Retrieval (search)

Another more data-driven approach to “listening” to your user is to search for previous statements in your logs of previous conversations. This is analogous to a human listener trying to recall where they’ve heard a question or statement or word before. A bot can search not only its own conversation logs, but also any transcript of human-to-human conversations, bot-to-human conversations, or even bot-to-bot conversations. But, as usual, garbage in means garbage out. So you should clean and curate your database of previous conversations to ensure that your bot is searching (and mimicking) high-quality dialog. You would like humans to enjoy the conversation with your bot.

###  Generative models

Sequence-to-sequence models are machine learning translation algorithms that “translate” statements by your user into replies by your chatbot.

If you want to build a creative chatbot that says things that have never been said before, generative models such as these may be what you need: 

* Sequence-to-sequence—Sequence models trained to generate replies based on their input sequences 
* Restricted Boltzmann machines (RBMs)—Markov chains trained to minimize an “energy” function 
* Generative adversarial networks (GANs)—Statistical models trained to fool a “judge” of good conversation 



### Pros and cons of each approach



| Approach   | Advantages                                                   | Disadvantages                                                |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Grammar    | Easy to get started <br />Training easy to reuse <br />Modular<br />Easily controlled/restrained | Limited “domain”<br />Capability limited by human effort <br />Difficult to debug <br />Rigid, brittle rules |
| Grounding  | Answers logical questions well <br />Easily controlled/restrained | Sounds artificial, mechanical <br />Difficulty with ambiguity <br />Difficulty with common sense <br />Limited by structured data <br />Requires large scale information extraction <br />Requires human curation |
| Retrieval  | Simple <br />Easy to “train” <br />Can mimic human dialog    | Difficult to scale <br />Incoherent personality <br />Ignorant of context <br />Can’t answer factual questions |
| Generative | New, creative ways of talking <br />Less human effort <br />Domain limited only by data <br />Context aware | Difficult to “steer” <br />Difficult to train <br />Requires more data (dialog) <br />Requires more processing to train |







## Sentiment



Whether you use raw single-word tokens, n-grams, stems, or lemmas in your NLP pipeline, each of those tokens contains some information. An important part of this information is the word’s sentiment—the overall feeling or emotion that the word invokes. This sentiment analysis—measuring the sentiment of phrases or chunks of text—is a common application of NLP. In many companies it’s the main thing an NLP engineer is asked to do. Companies like to know what users think of their products. So they often will provide some way for you to give feedback. A star rating on Amazon or Rotten Tomatoes is one way to get quantitative data about how people feel about products they’ve purchased. But a more natural way is to use natural language comments.

There are two approaches to sentiment analysis: 

* A rule-based (heuristics) algorithm composed by a human. A common rule-based approach to sentiment analysis is to find keywords in the text and map each one to numerical scores or weights in a dictionary.
* A machine learning model learned from data by a machine. Relies on a labeled set of statements or documents to train a machine learning model to create those rules.



### Sequence-to-sequence learning 

Sequence-to-sequence learning (often abbreviated as seq2seq learning) is a generalization of the sequence labeling problem. In seq2seq, Xi and Yi can have different lengths. seq2seq models have found application in machine translation (where, for example, the input is an English sentence, and the output is the corresponding French sentence), conversational interfaces (where the input is a question typed by the user, and the output is the answer from the machine), text summarization, spelling correction, and many others.

Many but not all seq2seq learning problems are currently best solved by neural networks. The network architectures used in seq2seq all have two parts: an encoder and a decoder. 

In seq2seq neural network learning, the encoder is a neural network that accepts sequential input. It can be an RNN, but also a CNN or some other architecture. The role of the encoder is to read the input and generate some sort of state (similar to the state in RNN) that can be seen as a numerical representation of the meaning of the input the machine can work with. The meaning of some entity, whether it be an image, a text or a video, is usually a vector or a matrix that contains real numbers. This vector (or matrix) is called in the machine learning jargon the embedding of the input.

The decoder is another neural network that takes an embedding as input and is capable of generating a sequence of outputs. As you could have already guessed, that embedding comes from the encoder. To produce a sequence of outputs, the decoder takes a start of sequence input feature vector x(0) (typically all zeroes), produces the first output y(1), updates its state by combining the embedding and the input x(0), and then uses the output y(1) as its next input x(1). 

More accurate predictions can be obtained using an architecture with **attention**. Attention mechanism is implemented by an additional set of parameters that combine some information from the encoder (in RNNs, this information is the list of state vectors of the last recurrent layer from all encoder time steps) and the current state of the decoder to generate the label. That allows for even better retention of long-term dependencies than provided by gated units and bidirectional RNN.





## Real-world NLP challenges





### Information extraction (named entity extraction and question answering)

Information extraction and question answering systems are used for 

* TA assistants for university courses 
* Customer service 
* Tech support 
* Sales 
* Software documentation and FAQs 

Information extraction can be used to extract things such as

* Dates 
* Times
* Prices 
* Quantities
* Addresses
* Names – People – Places – Apps – Companies – Bots 
* Relationships – “is-a” (kinds of things) – “has” (attributes of things) – “related-to”





#### Named entities and relations

You’d like your machine to extract pieces of information and facts from text so it can know a little bit about what a user is saying. A typical sentence may contain several named entities of various types, such as geographic entities, organizations, people, political entities, times (including dates), artifacts, events, and natural phenomena. And a sentence can contain several relations, too—facts about the relationships between the named entities in the sentence.

**A knowledge base.** 

('Stanislav Petrov', 'is-a', 'lieutenant colonel') This is an example of two named entity nodes ('Stanislav Petrov' and 'lieutenant colonel') and a relation or connection ('is a') between them in a knowledge graph or knowledge base. A collection of these triplets is a knowledge graph. This is also sometimes called an ontology by linguists, because it’s storing structured information about words. But when the graph is intended to represent facts about the world rather than merely words, it’s referred to as a knowledge graph or knowledge base.

A knowledge base can be used to build a practical type of chatbot called a question answering system (QA system). Customer service chatbots, including university TA bots, rely almost exclusively on knowledge bases to generate their replies.

**Information extraction.**

“Information extraction” is converting unstructured text into structured information stored in a knowledge base or knowledge graph. Information extraction is part of an area of research called natural language understanding (NLU), though that term is often used synonymously with natural language processing. Information extraction and NLU is a different kind of learning than you may think of when researching data science. It isn’t only unsupervised learning; even the very “model” itself, the logic about how the world works, can be composed without human intervention. Instead of giving your machine fish (facts), you’re teaching it how to fish (extract information). Nonetheless, machine learning techniques are often used to train the information extractor. 



#### Regular patterns



You need a pattern-matching algorithm that can identify sequences of characters or words that match the pattern so you can “extract” them from a longer string of text. The naive way to build such a pattern-matching algorithm is in Python, with a sequence of if/then statements that look for that symbol (a word or character) at each position of a string.

A pattern-matching engine is integrated into most modern computer languages, including Python. It’s called **regular expressions**. Regular expressions and string interpolation formatting expressions (for example, "{:05d}".format(42)), are mini programming languages unto themselves.

So regular expressions are the pattern definition language of choice for many NLP problems involving pattern matching. Regular expressions define a finite state machine or FSM—a tree of “if-then” decisions about a sequence of symbols. A finite state machine that operates on a sequence of symbols such as ASCII character strings, or a sequence of English words, is called a grammar. They can also be called formal grammars to distinguish them from natural language grammar rules you learned in grammar school.

**Information extraction as ML feature extraction.**

You want your machine learning pipeline to be able to do some basic things, such as answer logical questions, or perform actions such as scheduling meetings based on NLP instructions. And machine learning falls flat here. You rarely have a labeled training set that covers the answers to all the questions people might ask in natural language. Plus, as you’ll see here, you can define a compact set of condition checks (a regular expression) to extract key bits of information from a natural language string. And it can work for a broad range of problems.

**Pattern matching** (and regular expressions) continue to be the state-of-the art approach for information extraction. Even with machine learning approaches to natural language processing, you need to do feature engineering. You need to create bags of words or embeddings of words to try to reduce the nearly infinite possibilities of meaning in natural language text into a vector that a machine can process easily. Information extraction is just another form of machine learning feature extraction from unstructured natural language data, such as creating a bag of words, or doing PCA on that bag of words. 



### Information worth extracting 

##### Useful regular expression

Some keystone bits of quantitative information are worth the effort of “hand-crafted” regular expressions: 

* GPS locations 

  ```python
  # Regular expression for GPS coordinates
  
  import re
  lat = r'([-]?[0-9]?[0-9][.][0-9]{2,10})'
  lon = r'([-]?1?[0-9]?[0-9][.][0-9]{2,10})'
  sep = r'[,/ ]{1,3}'
  re_gps = re.compile(lat + sep + lon)
  
  re_gps.findall('http://...maps/@34.0551066,-118.2496763...')
  # [(34.0551066, -118.2496763)]
  re_gps.findall("https://www.openstreetmap.org/#map=10/5.9666/116.0566")
  # [('5.9666', '116.0566')]
  re_gps.findall("Zig Zag Cafe is at 45.344, -121.9431 on my GPS.")
  # [('45.3440', '-121.9431')]
  ```

* Dates 

  ```python
  # US Dates
  us = r'((([01]?\d)[-/]([0123]?\d))([-/]([0123]\d)\d\d)?)'
  mdy = re.findall(us, 'Santa came 12/25/2017. An elf appeared 12/12.')
  dates = [{'mdy': x[0], 'my': x[1], 'm': int(x[2]), 'd': int(x[3]),
            'y': int(x[4].lstrip('/') or 0), 'c': int(x[5] or 0)} for x in mdy]
  
  # European Dates
  eu = r'((([0123]?\d)[-/]([01]?\d))([-/]([0123]\d)?\d\d)?)'
  dmy = re.findall(eu, 'Alan Mathison Turing OBE FRS (23/6/1912-7/6/1954)\
  					was an English computer scientist.')
  
  # Recognizing month words
  mon_words = 'January February March April May June July ' \
  			'August September October November December'
  
  mon = (r'\b(' + '|'.join('{}|{}|{}|{}|{:02d}'.format(m, m[:4], m[:3], i + 1, i + 1) for i, m in
                           enumerate(mon_words.split())) + r')\b')
  
  re.findall(mon, 'January has 31 days, February the 2nd month of 12, has 28, except in a Leap Year.')
  
  # Validating Dates
  import datetime
  dates = []
  for g in groups:
  	month_num = (g['us_mon'] or g['eu_mon']).strip()
      
  	try:
  		month_num = int(month_num)
          
  	except ValueError:
  		month_num = [w[:len(month_num)]
                       for w in mon_words].index(month_num) + 1
  	
      date = datetime.date(
          int(g['us_yr'] or g['eu_yr']),
          month_num,
          int(g['us_day'] or g['eu_day']))
  	
      dates.append(date)
  ```

  

* Prices 

* Numbers 

Other important pieces of natural language information require more complex patterns than are easily captured with regular expressions:

* Question trigger words 
* Question target words 
* Named entities





### Extracting relationships (relations)



Extracting the dates and the GPS coordinates might enable you to associate that date and location with Desoto, the Pascagoula people, and two rivers whose names you can’t pronounce. You’d like your bot (and your mind) to be able to connect those facts to larger facts—for example, that Desoto was a Spanish conquistador and that the Pascagoula people were a peaceful Native American tribe. And you’d like the dates and locations to be associated with the right “things”: Desoto, and the intersection of two rivers, respectively.



