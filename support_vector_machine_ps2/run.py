import pylab

# Run Script for ps1 (Perceptron Implementation)

# Part 1: Find the optimal hyperplane for the hard margin support vector machine problem
# Plot points
class_1 = [(2,2), (4,4), (4,0)]
(class_1_x, class_1_y) = zip(*class_1)

class_minus_1 = [(0,0), (2,0), (0,2)]
(class_minus_1_x, class_minus_1_y) = zip(*class_minus_1)

pylab.plot(class_1_x, class_1_y, 'bo')
pylab.plot(class_minus_1_x, class_minus_1_y, 'ro')

# Plot hyperplane
hyperplane = [(4,-1), (3,0), (2,1), (1,2), (0,3), (-1,4)]
(hyperplane_x, hyperplane_y) = zip(*hyperplane)

# Plot weight vector
pylab.arrow(1.5, 1.5, 0.3, 0.3, head_width=0.05, head_length=0.1)

pylab.plot(hyperplane_x, hyperplane_y)

pylab.axis('scaled')
pylab.xlim((-0.1,5.1))
pylab.ylim((-0.1,5.1))
pylab.legend(['Elements in Class: +1', 'Elements in Class: -1'])

pylab.grid(True)
pylab.savefig("SVM_Plot.png")
pylab.show()

print('\n=========================================================================================')
print('Script complete')
