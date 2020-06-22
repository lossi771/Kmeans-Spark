import sys, getopt
from pyspark import SparkContext
import numpy as np
import random

threshold = 0.01

def parseInputs(argv):
  try:
    opts, args = getopt.getopt(argv,"k:i:o:m:h",["clusters=","input_file=", "output_file=", "max_iter="])
  except getopt.GetoptError:
      print ('kmeans.py -k <clusters> -i <input_file> -o <output_file> -m <max_iter>')
      sys.exit(2)
  
  for opt, arg in opts:
      max_iter = 15
      if opt == '-h':
         print ('kmeans.py -k <clusters> -i <input_file> -o <output_file> -m <max_iter>')
         print ('or')
         print ('generate_point.py --clusters <clusters> --input_file <input_file> --output_file <output_file> --max_iter <max_iter>')
         sys.exit()
      elif opt in ("-k", "--clusters"):
        clusters = int(arg)
      elif opt in ("-i", "--input_file"):
        input_points = arg
      elif opt in ("-o", "--output_file"):
        output_centroids = arg
      elif opt in ("-m", "--max_iter"):
        max_iter = int(arg)
  return (clusters, input_points, output_centroids, max_iter)

def create_point(line_of_coordinates):
    return np.fromstring(line_of_coordinates, dtype = np.float32, sep=' ')

def generate_centroids(number_of_clusters, points):
    array_of_points = points.collect()
    np_centroids = random.sample(list(array_of_points), number_of_clusters)
    centroids = [(i, p.tolist()) for p, i in zip(np_centroids, range(1, number_of_clusters + 1)) ] # lista di liste ******* INDICE CENTROIDI PARTE DA 1 **********
    return centroids

def assign_to_closest_mean(point):
  centroids = broadcasted_centroids.value
  centroids_array = np.array(centroids, dtype = float)
  distance = np.inf
  min_mean_id = -1 # indice del mean con distanza minima 
  for current_mean_index in range(len(centroids_array)):
    current_distance = np.linalg.norm(point - centroids_array[current_mean_index, 1])
    if(current_distance < distance):
      distance = current_distance
      min_mean_id = centroids_array[current_mean_index][0]
  return (min_mean_id, point)

def update_centroids(mean_pointlist):
  arrayPoints = np.stack(list(mean_pointlist[1]), axis = 0) #rivedere per bene
  return (mean_pointlist[0], np.mean(arrayPoints, axis = 0))

def threshold_check(new_centroids, centroids):
  if len(new_centroids) != len(centroids):
    print("centroid array sizes are not equal !")
    sys.exit(-1)
  return (max ([np.linalg.norm(np.array(new_centroids[i][1]) - np.array(centroids[i][1])) for i in range(len(centroids))]) > threshold )


if __name__ == "__main__":
    number_of_clusters, input_points, output_centroids, max_iter = parseInputs(sys.argv[1:])
    master = "local" #cambiare quando si carica su cluster in "yarn"
    #sc = SparkContext(master, "KMeans")
    sc = SparkContext(appName="KMeans-App", master = "local[*]")
    points = sc.textFile(input_points).map(create_point).cache()
    centroids = generate_centroids(number_of_clusters, points)
    
    converged = False
    current_iteration = 0

    while not converged: # convergenza sul massimo numero di iterazioni e sul fatto che i centroidi non cambiano più?
      broadcasted_centroids = sc.broadcast(centroids)
      
      mean_pointlist = points.map(assign_to_closest_mean).groupByKey() # cid-iterable di punti assegnati, shuffling 
      new_centroids = mean_pointlist.map(update_centroids).collect()
      # mettere in documentazione il fatto che il numero dei cluster assumiamo sia piccolo e che quindi 
      # non vada calcolato il conteggio dello spostamento dei centroidi in modo distribuito

      converged = (current_iteration == max_iter - 1) or threshold_check(new_centroids, centroids)
      current_iteration += 1
      centroids = new_centroids
    
    with open(output_centroids, "w") as file: 
      for centroid in centroids:
        file.write(str(centroid[0]) + " ".join(str(round(i, 3)) for i in centroids))
        file.write('\n')