import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

#vector and counting
v1 = np.random.randint(0, 20, 100)
unique, counts = np.unique(v1, return_counts=True)

#ploting
plt.bar(unique, counts)
plt.xticks(unique)
plt.xlabel("liczby")
plt.ylabel("czestotliwosc")
plt.suptitle("Czestosc")
plt.show()
