import sys
import codecs
import ujson
import itertools
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

def preprocess_tweets(tweets):
    stemmer = SnowballStemmer("english")
    stop = stopwords.words("english")
    tweet_texts = [ " ".join([ stemmer.stem(i)
                                for i in tweet["text"].lower().split()
                                if i not in stop ])
                    for tweet in tweets ]
    return tweet_texts

def get_tfidf_vectors(tweets):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(tweets)

def cluster_DBSCAN(tweets_tfidf_matrix, minPts, eps):
    clusterer = DBSCAN(eps=eps, min_samples=minPts, metric="cosine", algorithm="brute")
    return clusterer.fit_predict(tweets_tfidf_matrix)

def main():
    if len(sys.argv) == 4:
        try:
            minPts = int(sys.argv[2])
            eps = float(sys.argv[3])

            with codecs.open(sys.argv[1], "r", "utf-8") as input_file:
                tweets = [ ujson.loads(line) for line in input_file ]
                print("Input data read")

            tweets_preprocessed = preprocess_tweets(tweets)
            tweets_tfidf_matrix = get_tfidf_vectors(tweets_preprocessed)
            cluster_labels = cluster_DBSCAN(tweets_tfidf_matrix, minPts, eps)

            nclusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print("%d clusters present in dataset" % nclusters)

            for cluster_label, cluster_tweets in itertools.groupby(zip(tweets, cluster_labels),
                                                                    lambda x: x[1]):
                print("\nCluster", "Noise" if cluster_label == -1 else cluster_label)
                for cluster_tweet in cluster_tweets:
                    print(cluster_tweet[0]["text"])
        except ValueError:
            print("Invalid input values")
        except IOError:
            print("Input file does not exist")
    else:
        print("Usage: python %s file minPts epsilon" % sys.argv[0])

if __name__ == "__main__":
    main()

