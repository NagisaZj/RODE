import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

color = {'coor_t': [159, 67, 255],  # Purple
         0: [216, 30, 54],  # Red
         'full_action_space': [204, 0, 204],  # Magenta
         1: [254, 151, 0],  # Orange
         'plus_r': [0, 0, 0],  # Black background
         'h_resac': [180, 180, 180],  # Grey board walls
         2: [0, 255, 0],  # Green apples
         '7': [255, 255, 0],  # Yellow fining beam
         '9': [159, 67, 255],  # Purple player

         # Colours for agents. R value is a unique identifier
         3: [2, 81, 154],  # Blue
         'q-q': [100, 255, 255],  # Cyan
         'cen': [83, 134, 139],  # Lavender
         'plus_v': [250, 204, 255],  # Pink
         'cen_control': [238, 223, 16],
         'social_influence': [238, 232, 170]}
color = {key: np.array(value) / 255. for key, value in color.items()}

path = './results/pic_replays'
files = []

for r, d, f in os.walk(path):
    for file in f:
        if file == 'role_frequency.pkl':
            files.append(os.path.join(r, file))

files.sort()
files = files[1:]

fre = []

for file in files:
    with open(file, "rb") as f:
        data = pickle.load(f)

    fre.append(data)

fre = np.array(fre)
hbos = np.linspace(100, 5200, 100)

figure = plt.figure()
for i in range(fre.shape[1]):
    plt.plot(fre[:, i], c=color[i])

plt.legend([str(i) for i in range(fre.shape[1])])
plt.title('Role Frequency')
plt.savefig('role_frequency.png')
plt.close()
