from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import csv,re,sys,spacy
import numpy as np
from string import punctuation,printable


'''Creating a Vector Feature Matrix with numpy'''
# vocab of unique words over entire corpus
vocab_set = set()
[[vocab_set.add(token) for token in tokens] for tokens in my_docs]
vocab = list(vocab_set)

# reverse lookup - key/word, value/list_idx
vocab_dict = {word: i for i, word in enumerate(vocab)}

# word count vector maxtrix
# row/doc, column/word
word_counts = np.zeros((len(docs), len(vocab)))
for doc_id, words in enumerate(my_docs):
    for word in words:
        word_id = vocab_dict[word]
        word_counts[doc_id][word_id] += 1

# document frequencies
df = np.sum(word_counts > 0, axis=0)

# normalize word count matrix to get term frequencies
tf_norm = np.sqrt((word_counts ** 2).sum(axis=1))
tf_norm[tf_norm == 0] = 1
tf = word_counts / tf_norm.reshape(len(my_docs), 1)
# tf.shape = (999,27446)

# TF-IDF is then calculated by multiplying the Term-Frequency by the Inverse-Document-Frequency
idf = np.log((len(my_docs) + 1.) / (1. + df)) + 1.
tfidf = tf * idf

# normalize the tf-idf matrix
tfidf_norm = np.sqrt((tfidf ** 2).sum(axis=1))
tfidf_norm[tfidf_norm == 0] = 1
tfidf_normed = tfidf / tfidf_norm.reshape(len(my_docs), 1)


'''Term Frequency Inverse Document Frequency'''
c_train = [’Here is my corpus of text it says stuff and things’, ’Here is some other document’]
c_test = [’Yet another document’, ’This time to test on’]
# Converts a collection of raw documents to a matrix of TF-IDF features
tfidf = TfidfVectorizer(stop_words='english') 
tfidf.fit(c_train)
test_arr = tfidf.transform(c_test).todense() # transform sparse matrix
# Print out the feature names
print(tfidf.get_feature_names())

def tokenize(doc):
    '''Tokenize and Stem'''
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]

# Convert a collection of text documents to a matrix of token counts
count_vectorized = countvect.fit_transform(documents)
countvect = CountVectorizer(tokenizer=tokenize, stop_words='english',\
        ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, binary=False,)
print("Count of 'dinner':", count_vectorized[0, words.index('dinner')])

'''Find cosine similarity between documents'''
documents = ['Dogs like dogs more than cats.',
             'The dog chased the bicycle.',
             'The cat rode in the bicycle basket.',
             'I have a fast bicycle.']
sbs = SnowballStemmer('english')
punctuation = set(punctuation)
def my_tokenizer(text):
    return [sbs.stem(token) for token in word_tokenize(text) if token not in punctuation]
vectorizer = TfidfVectorizer(tokenizer=my_tokenizer, stop_words='english',\
        ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, binary=False,)
tfidf_docs = vectorizer.fit_transform(documents)
cos_sims = linear_kernel(tfidf_docs, tfidf_docs) # try cosine_similarity
print(cos_sims) # highest value is most similiar


'''spaCy object'''
if not 'nlp' in locals():
    print("Loading English Module...")
    nlp = spacy.load('en')

doc = nlp("And what would you do if you met a jaboo?")
lemmatized_tokens = [token.lemma_ for token in doc] # token.text for un-lemmatized word
print(lemmatized_tokens)

def sentence_list(doc):
    '''Extract sentences from a spaCy document object'''
    sents = []
    # The sents attribute returns spans which have indices into the original spacy.tokens.Doc.
    # Each index value represents a token
    for span in doc.sents:
        sent = ’’.join(doc[i].string for i in range(span.start,span.end)).strip()
        sents.append(sent)
    return sents

def lemmatize_string(doc, stop_words):
    # remove punctuation from string
    if sys.version_info.major == 3:
        PUNCT_DICT = {ord(punc): None for punc in punctuation}
        doc = doc.translate(PUNCT_DICT)
    else:
        doc = unicode(doc.translate(None, punctuation)) # spaCy expects a unicode object
    # remove unicode, clean, and lemmatize
    clean_doc = "".join([c for c in doc if c in printable])
    doc = nlp(clean_doc)
    tokens = [re.sub("\W+","",t.lemma_.lower()) for t in doc]
    return ’ ’.join(w for w in tokens if w not in stop_words)

if __name__=="__main__":
    corpus = ["oh the thinks you can think if only you try"]
    STOPLIST = set(list(ENGLISH_STOP_WORDS) + ["n’t", "’s", "’m"])
    processed = [lemmatize_string(doc, STOPLIST) for doc in corpus]


'''Extract Similar Words Using Vector Representation (word2vec)'''
def get_similar_words(wrd, top_n=10):
    token = nlp(wrd)[0]
    if not token.has_vector:
        raise ValueError("{} doesn’t have a vector representation".
    format(wrd))
    cosine = lambda v1, v2: np.dot(v1, v2) / (norm(v1) * norm(v2))
    # Gather all words spaCy has vector representations for
    all_words = list(w for w in nlp.vocab if w.has_vector
    and w.orth_.islower() and w.lower_ != token.lower_)
    # Sort by similarity to token
    all_words.sort(key=lambda w: cosine(w.vector, token.vector))
    all_words.reverse()
    print("Top {} most similar words to {}:".format(top_n, token))
    for word in all_words[:top_n]:
        print(word.orth_, "\t", cosine(word.vector, token.vector))
