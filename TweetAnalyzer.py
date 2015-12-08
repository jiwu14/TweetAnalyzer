import sys
import codecs
import ujson
import itertools
import string
from operator import itemgetter
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def preprocess_tweets(tweets):
    stemmer = SnowballStemmer("english")
    stop = set(stopwords.words("english"))
    tweet_texts = [ " ".join(stemmer.stem(i) if len(i) > 1 else i
                                for i in ("".join(c for c in word if c not in string.punctuation)
                                            for word in tweet["text"].lower().split())
                                if i and i not in stop)
                    for tweet in tweets ]
    return list(set(tweet_texts))

def get_sparse_dist_matrix(tweets_tfidf_matrix, eps):
    """Get the sparse distance matrix from the pairwise cosine distance
    computations from the given tfidf vectors. Only distances less than or
    equal to eps are put into the matrix"""
    rows = []
    cols = []
    data = []
    for ndx, tweet in enumerate(tweets_tfidf_matrix):
        rows.append(len(cols))
        distances = cosine_distances(tweet, tweets_tfidf_matrix)[0]
        for other_ndx, dist in enumerate(distances):
            if ndx != other_ndx and dist <= eps:
                cols.append(other_ndx)
                data.append(dist)
    return csr_matrix((data, cols, rows), dtype=int)

def get_tfidf_vectors(tweets):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(tweets)

def cluster_DBSCAN(data, minPts, eps, sparse_dist=False):
    clusterer = DBSCAN(eps=eps, min_samples=minPts, metric="precomputed" if sparse_dist
                                                    else "cosine", algorithm="brute")
    return clusterer.fit_predict(data)

def cluster_agglomerative(data, nclusters):
    clusterer = AgglomerativeClustering(n_clusters=nclusters, affinity="cosine", linkage="complete")
    return clusterer.fit_predict(data)

def main():
    if len(sys.argv) == 4 or len(sys.argv) == 5:
        try:
            minPts = int(sys.argv[2])
            eps = float(sys.argv[3])
            sparse_dist = len(sys.argv) == 5 and sys.argv[4] == "sparse"

            with codecs.open(sys.argv[1], "r", "utf-8") as input_file:
                tweets = [ ujson.loads(line) for line in input_file ]
                print("%d json objects read" % len(tweets))

            tweets_preprocessed = preprocess_tweets(tweets)
            print("Preprocessed %d unique tweets" % len(tweets_preprocessed))
            tweets_tfidf_matrix = get_tfidf_vectors(tweets_preprocessed)
            print("Tfidf vectors generated")

            if sparse_dist:
                tweets_sparse_dist_matrix = get_sparse_dist_matrix(tweets_tfidf_matrix, eps)
                print("Sparse distance matrix generated")

            cluster_labels = cluster_DBSCAN(tweets_sparse_dist_matrix if sparse_dist
                                            else tweets_tfidf_matrix, minPts, eps, sparse_dist)
            print("Clustered with DBSCAN")

            nclusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print("%d clusters present in dataset" % nclusters)

            #cluster_labels = cluster_agglomerative(tweets_tfidf_matrix.toarray(), nclusters + (1 if -1 in cluster_labels else 0))
            #s_score = silhouette_score(tweets_tfidf_matrix, cluster_labels, metric="cosine")
            #print("silhouette score: %f" % s_score)

            clustered_tweets_grouped = ((cluster, list(cluster_t))
                                        for cluster, cluster_t in
                                            itertools.groupby(sorted(
                                                                zip(tweets_preprocessed, cluster_labels),
                                                                key=itemgetter(1)),
                                                            lambda x: x[1]))

            for cluster_label, cluster_tweets in itertools.islice(clustered_tweets_grouped, 1, None):
                print("\nCluster", "Noise" if cluster_label == -1 else cluster_label)
                for cluster_tweet in cluster_tweets:
                    print(cluster_tweet[0])
        #except ValueError:
        #    print("Invalid input values")
        except IOError:
            print("Input file does not exist")
    else:
        print("Usage: python %s file minPts epsilon [sparse]\n\"sparse\" - calculate a sparse distance matrix instead of using numpy" % sys.argv[0])

if __name__ == "__main__":
    main()

