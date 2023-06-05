import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml

map_id = np.random.randint(1, 50)

map_png = f"tracks/maps/map{map_id}.png"
map_csv = f"tracks/centerline/map{map_id}.csv"
map_yaml = f"tracks/maps/map{map_id}.yaml"

image = mpimg.imread(map_png)
lines = open(map_csv, 'r').readlines()

map_data = [line.strip().split(',')[1:3] for line in lines[1:]]
map_data = np.array(map_data, dtype=float)

with open(map_yaml, "r") as file:
    yaml_data = yaml.safe_load(file)

map_origin = yaml_data["origin"][0:2]
map_resolution = yaml_data["resolution"]

map_scaled = (map_data - map_origin) / map_resolution

flipped_image = np.flipud(image)
plt.imshow(flipped_image)
plt.plot(map_scaled[:, 0], map_scaled[:, 1], markersize=1)
plt.show()
