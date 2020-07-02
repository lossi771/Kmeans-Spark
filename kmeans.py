import sys, getopt
from pyspark import SparkContext
import numpy as np
import random

BENCHMARK = False
threshold = 0.1
max_iter = 15

def parseInputs(argv):
  global threshold, max_iter, BENCHMARK
  try:
    opts, args = getopt.getopt(argv,"k:i:t:o:m:h",["clusters=","input_file=", "output_file=", "max_iter=", "threshold="])
  except getopt.GetoptError:
      print ('[USAGE] kmeans.py -k <clusters> -i <input_file> -o <output_file> -m <max_iter> -t <threshold>')
      sys.exit(2)
  if(len(opts) != 5):
    print(len(argv))
    print ('[USAGE] kmeans.py -k <clusters> -i <input_file> -o <output_file> -m <max_iter> -t <threshold>')
    sys.exit(2)

  for opt, arg in opts:
      if opt == '-h':
         print ('[USAGE] kmeans.py -k <clusters # or file> -i <input_file> -o <output_file> -m <max_iter>')
         print ('or')
         print ('[USAGE] generate_point.py --clusters <clusters # or file> --input_file <input_file> --output_file <output_file> --max_iter <max_iter>')
         sys.exit()
      elif opt in ("-k", "--clusters"):
        if(arg.isnumeric()): 
          clusters = int(arg)
        else:
          clusters = arg
          BENCHMARK = True
      elif opt in ("-i", "--input_file"):
        input_points = arg
      elif opt in ("-o", "--output_file"):
        output_centroids = arg
      elif opt in ("-m", "--max_iter"):
        max_iter = int(arg)
      elif opt in ("-t", "--threshold"):
        threshold = float(arg)
  return (clusters, input_points, output_centroids)

def create_point(line_of_coordinates):
    return np.fromstring(line_of_coordinates, dtype = np.float32, sep=' ')

def generate_centroids(number_of_clusters, points):
    array_of_points = points.collect()
    np_centroids = random.sample(list(array_of_points), number_of_clusters) # get k cluster from the points
    centroids = [(i, p) for p, i in zip(np_centroids, range(1, number_of_clusters + 1)) ] # list of (id, np.array(point))
    return np.array(centroids, dtype=object) 

def get_centroids_from_file(path):
  centroids = []
  with open(path, "r") as file:
    for line in file.readlines():
      label, coord = line.split("\t")
      centroids.append((label, np.fromstring(coord, dtype = np.float32, sep=' ')))
  return np.array(centroids, dtype = object)


def assign_to_closest_mean(point):
  centroids_array = broadcasted_centroids.value
  distance = np.inf
  min_mean_id = -1 # indice del mean con distanza minima 
  for current_mean_index in range(len(centroids_array)):
    current_distance = np.linalg.norm(point - centroids_array[current_mean_index][1])
    if(current_distance < distance):
      distance = current_distance
      min_mean_id = centroids_array[current_mean_index][0]
  return (min_mean_id, point)

def update_centroids(mean_pointlist):
  arrayPoints = np.stack(list(mean_pointlist[1]), axis = 0) # Join a sequence of arrays along a new axis.
  return (mean_pointlist[0], np.mean(arrayPoints, axis = 0))

def threshold_check(new_centroids, centroids):
  if len(new_centroids) != len(centroids):
    print("centroid array sizes are not equal !")
    sys.exit(-1)
  distances = [np.linalg.norm(np.array(new_centroids[i][1]) - np.array(centroids[i][1])) for i in range(len(centroids))]
  print("Converged centroids: {} of {}".format(sum(i <= threshold  for i in distances), len(centroids)))
  return (max(distances) <= threshold )


if __name__ == "__main__":
    clusters, input_points, output_centroids = parseInputs(sys.argv[1:])
    sc = SparkContext(appName="KMeans-App", master = "yarn")
    sc.setLogLevel("WARN")
    points = sc.textFile(input_points).map(create_point).cache()

    if BENCHMARK:
      centroids = get_centroids_from_file(clusters)
    else:
      centroids = generate_centroids(clusters, points)
        
    converged = False
    current_iteration = 1

    while not converged: # criteria: max iteration or threshold
      print("curr_iteration: " + str(current_iteration))
      broadcasted_centroids = sc.broadcast(centroids)
      mean_pointlist = points.map(assign_to_closest_mean).groupByKey()
      new_centroids = mean_pointlist.map(update_centroids).sortByKey(ascending=True).collect()
     
      converged = (current_iteration == max_iter - 1) or threshold_check(new_centroids, centroids)
     
      current_iteration += 1
      centroids = new_centroids

    with open(output_centroids, "w") as file: 
      for centroid in centroids:
        file.write(str(centroid[0])+ "\t" + " ".join(str(round(i, 3)) for i in centroid[1]) + "\n")