import csv
import numpy as np
from numpy import argmax
from numpy.linalg import norm


DIMENSION = 100
TOTAL = 50000
CATEGORY = 5
THRESHOLD = 100

RANDOM_INIT_MEANS = False
# random initialing is faster but may fall into error
# while dispersed initialing costs time but has a stable performance.

data = np.zeros((TOTAL, DIMENSION))
means = np.zeros((CATEGORY, DIMENSION))
label = np.zeros(TOTAL, dtype=int)

def getData():
    global data
    with open("../data/features.csv", 'r') as csvFile:
        csvFile.readline()
        csv_reader = csv.reader(csvFile)
        PID = 0
        for row in csv_reader:
            data[PID] = row[1:] 
            PID += 1

def initRandomMeans():
    global means
    choice = np.random.choice(TOTAL, CATEGORY, replace=False)
    means = data[choice]

def initDispersedMeans():
    global means
    choice = np.zeros(CATEGORY, dtype=int)
    choice[0] = np.random.randint(TOTAL)
    distance2Means = np.full(50000, np.infty)
    for i in range(CATEGORY-1):
        for candidate in range(TOTAL):
            distance2Means[candidate] = min(distance2Means[candidate], norm(data[candidate] - data[choice[i]]))
        choice[i+1] = argmax(distance2Means)
    means = data[choice]

def initMeans():
    if RANDOM_INIT_MEANS:
        initRandomMeans()
    else:
        initDispersedMeans()

def storeResult(filename, labelMap):
    with open(filename, 'w') as output:
        output.write("id,category\n")
        for i in range(TOTAL):
            output.write("{},{}\n".format(i, labelMap[label[i]]))

def mapLabelName():
    ### return the map: current cluster ID -> radius-sorted cluster ID
    radius = np.zeros(CATEGORY)
    ### TODO: calculate radius of each cluster and store them in var:radius ###
    global label
    for i in range(CATEGORY):
        maxradius = 0
        for index in range(TOTAL):
            if label[index] == i:
                rad = euclDistance(data[index],means[i])
                if rad >= maxradius:
                    maxradius = rad
        radius[i] = maxradius
    print(radius)
    ### end of TODO ###
    temp =  radius.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(CATEGORY)
    print("Radius: ")
    print(radius[temp])
    return ranks

### TODO ###
### you can define some useful function here if you want
# this function is used to compute the distance of two data
def euclDistance(x1, x2):
    return np.sqrt(sum((x2 - x1) ** 2))

# 传入数据集和k值
def kmeans(dataset, k ,centroid):
    clusterData = np.array(np.zeros((TOTAL, 2),dtype=int))
    clusterChanged = True
    centroids = np.copy(centroid)
    while clusterChanged:
        clusterChanged = False
        for i in range(TOTAL):
            minDist = 100000.0
            minIndex = 0
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataset[i, :])
                if distance < minDist:
                    minDist = distance
                    clusterData[i, 1] = minDist
                    minIndex = j
            if clusterData[i, 0] != minIndex:
                clusterChanged = True
                clusterData[i, 0] = minIndex
        for j in range(k):
            cluster_index = np.nonzero(clusterData[:, 0] == j)
            pointsInCluster = data[cluster_index]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)
    return centroids, clusterData

### end of TODO ###
        

def main():
    getData()
    initMeans()
    ### TODO ###
    global label
    global means
    global data
    # implement your clustering alg. here
    centroids,clusterData = kmeans(data,CATEGORY,means)
    means = centroids
    label = clusterData[:,0]
    print(label)
    ### end of TODO ###
    labelMap = mapLabelName()
    storeResult("../data/predictions.csv", labelMap)


if __name__ == "__main__":
    main()
        
        