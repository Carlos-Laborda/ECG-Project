import numpy as np
import matplotlib.pyplot as plt

participant = np.load("../data/interim/30106/high_physical_activity.npy")
plt.plot(participant[1000:2000])

participant = np.load("../data/interim/30106/baseline.npy")
plt.plot(participant)

participant = np.load("../data/interim/30106/mental_stress.npy")
plt.plot(participant[:1000])
