import Tkinter
import matplotlib.pyplot as plt

execfile('mnist_load.py')
execfile('phi.py')
execfile('SDH.py')

plt.plot([16,32,64,96,128], [0.8033,0.8034,0.8037,0.8056,0.8053])
plt.ylabel('Precision')
plt.xlabel('Code Length')
plt.show()

plt.plot([16,32,64,96,128], [0.8052,0.7792,0.7571,0.5750,0.4252])
plt.ylabel('Recall')
plt.xlabel('Code Length')
plt.show()
