import sys, getopt
from pyspark import SparkContext
import numpy as np

def parseInputs(argv):
  try:
    opts, args = getopt.getopt(argv,"k:i:o:h",["clusters=","input_file=", "output_file="])
  except getopt.GetoptError:
      print ('kmeans.py -k <clusters> -i <input_file> -o <output_file>')
      sys.exit(2)
  
  for opt, arg in opts:
      if opt == '-h':
         print ('kmeans.py -k <clusters> -i <input_file> -o <output_file>')
         print ('or')
         print ('generate_point.py --clusters <clusters> --input_file <input_file> --output_file <output_file>')
         sys.exit()
      elif opt in ("-k", "--clusters"):
        clusters = int(arg)
      elif opt in ("-i", "--input_file"):
        input_points = arg
      elif opt in ("-o", "--output_file"):
        output_centroids = arg
  return (clusters, input_points, output_centroids)

def create_point(line_of_coordinates):
    return np.fromString(line_of_coordinates, dtype = np.float32, sep=' ')

def generate_centroids(number_of_clusters, points):
    array_of_points = points.collect()
    np_centroids = random.sample(list(array_of_points), number_of_clusters)
    centroids = [p.tolist() for p in np_centroids] # lista di liste
    return centroids
    

if __name__ == "__main__":
    number_of_clusters, input_points, output_centroids = parseInputs(sys.argv[1:])
    master = "local" #cambiare quando si carica su cluster in "yarn"
    sc = SparkContext(master, "KMeans")
    points = sc.textFile(input_points).map(create_point).cache()
    centroids = generate_centroids(number_of_clusters, points)
    
    converged = False

    while not converged:
        