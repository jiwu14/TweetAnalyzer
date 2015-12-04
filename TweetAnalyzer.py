import sys
import codecs
import ujson
from sklearn.cluster import DBSCAN

def main():
    if len(sys.argv) == 4:
        try:
            minPts = int(sys.argv[2])
            eps = float(sys.argv[3])

            with codecs.open(sys.argv[1], "r", "utf-8") as input_file:
                tweets = [ ujson.loads(line) for line in input_file ]
                print("Input data read")
        except ValueError:
            print("Invalid input values")
        except IOError:
            print("Input file does not exist")
    else:
        print("Usage: python %s file minPts epsilon" % sys.argv[0])

if __name__ == "__main__":
    main()

