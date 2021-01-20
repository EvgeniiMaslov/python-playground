import re
import string
from spellchecker import SpellChecker
from nltk.corpus import stopwords
import operator
import numpy as np

STOPWORDS = set(stopwords.words('english'))

def remove_digits(text):
    return re.sub(r'\d+', '', text)
    

def remove_url(text):
    url = re.compile(r'https?:\/\/t.co\/[A-Za-z0-9]+')
    return url.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'',text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_punctuation(text):
    text = text.translate(str.maketrans("","", string.punctuation))
    return text


spell = SpellChecker()
def spell_correction(text):
    
    corrected_text = []

    misspelled = spell.unknown(text.split())
    
    for word in text.split():
        if word in misspelled:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    
    return ' '.join(corrected_text)



def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]


def build_vocab(X):
    
    tweets = X.apply(lambda x: x.lower().split()).values
    vocab = {}
    
    for tweet in tweets:
        for word in tweet:
            count = vocab.get(word, 0)
            vocab[word] = count + 1
    return vocab


def check_embedding_coverage(X, embeddings):
    
    vocab = build_vocab(X)
    
    covered = {}
    oov = {}    
    n_covered = 0
    n_oov = 0
    
    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]
            
    vocab_coverage = len(covered) / len(vocab)
    text_coverage = (n_covered / (n_covered + n_oov))
    
    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_oov, vocab_coverage, text_coverage



def read_glove_embed(filepath):
    embeddings_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict