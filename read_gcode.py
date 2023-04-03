#!/usr/bin/python3

import os
import matplotlib.pyplot as plt
import time



# sys.path.append(os.path.abspath(".."))

import gcodeparser as gcp

start = time.time()
model = gcp.parse_gcode(os.path.abspath("zy322.gcode"), 5, 10, "Min-max")
# print("Execution time: {} seconds".format(time.time()-start))
layer=model.layers
Maxz=model.max_z
# print(Maxz) 
Maxz=16.4/0.2
N=[]
for i in range(1,3):#int(Maxz)):
    if i % 1 == 0:
        N.append(i)
# print(N)
# A=layer.gen_sample_points("Min-max",layer,10,0)
# print(A)
# for j in N:
#     Height=0.2*j
#     j=int(j)
#     print(Height)
#     layer = model.layers[j]

#     layer.to_svg(model.max_y, model.max_x, 'test.svg')

#     layer.plot_layer('k', 'b')
#     # plt.show()
#     plt.title('Layer %d'%j)
#     layers=model.layers
#     A=layer.gen_sample_points("Min-max",layers,10,0)
#     B=A[0]
#     C=A[1]




j=50
Height=0.2*j
print(Height)
layer = model.layers[j]

layer.to_svg(model.max_y, model.max_x, 'test.svg')

layer.plot_layer('k', 'b')
# plt.show()
plt.title('Layer %d'%j)
layers=model.layers
A=layer.gen_sample_points("Min-max",layers,50,0,4)
B=A[0]
C=A[1]
print("G01 X%f Y%f Z%f" % (B[0],C[0],Height))

"""
AlphaShape test here
"""
# Put all points in layer into a numpy array
# points = layer.get_points()
# point_list = []
# [point_list.append([x,y]) for x,y in zip(points['x'], points['y'])]
# features = np.array(point_list)

# # Import Delaunay and ConvexHull functions from scipy
# from scipy.spatial import Delaunay, ConvexHull

# # Create the Delaunay triangulation
# tri = Delaunay(features)

# valid_simplices = []
# bound_simplices = []

# # Alpha value of 0.8mm
# alpha = 0.8

# # Iterate through simplices in the triangulation
# for i, element in enumerate(tri.simplices):
#     # Check the length of each edge of the simplex 
#     l1 = np.sqrt(np.sum((features[element[0], :] - features[element[1], :])**2))
#     l2 = np.sqrt(np.sum((features[element[0], :] - features[element[2], :])**2))
#     l3 = np.sqrt(np.sum((features[element[1], :] - features[element[2], :])**2))
#     num_neighbors = np.sum(tri.neighbors[i] != -1)
#     # If all the edges are smaller than alpha, then the simplex is part of the body
#     if l1 < alpha and l2 < alpha and l3 < alpha:
#         valid_simplices.append(i)

# # Iterate through each valid simplex, checking for the number of neighbors it has
# for i, neighbors in enumerate(tri.neighbors[valid_simplices]):
#     # If a simplex has less than 3 layers, it is on a boundary
#     if np.sum(np.isin(valid_simplices, neighbors)) < 3:
#         bound_simplices.append(valid_simplices[i])

# plt.triplot(features[:,0], features[:,1], tri.simplices.copy()[bound_simplices,:])
# plt.show()

