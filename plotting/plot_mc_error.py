import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import colormaps

matplotlib.rcParams.update({'font.size': 14})
with open('./data/plotmcerror.json') as f:
    data = json.load(f)

cmap = matplotlib.cm.get_cmap("plasma")

# plot sinodal curve
min_position = -1.2
max_position = 0.6

xs = np.linspace(min_position, max_position, 100)
ys = np.sin(3 * xs) * 0.45 + 0.55

plt.figure(figsize=(10,6))
# contour of  mountain
plt.plot(xs, ys, label="Map Contour", color="grey")

# plots error directional
x_pos = []
# is target - pred
# target is 1 pred is 0.9 -> error 0.1
x_pos_error = []
for pos, err in zip(data["state"], data["error"]):
    x_pos.append(pos[0])
    x_pos_error.append(err[0])
print(len(x_pos))
print(len(x_pos_error))
print(len(np.sin(3 * np.array(x_pos)) * 0.45 + 0.55))
norm = matplotlib.colors.Normalize(vmin=min(-np.array(x_pos_error)), vmax=max(-np.array(x_pos_error)))
plt.bar(x_pos, -np.array(x_pos_error), bottom=np.sin(3 * np.array(x_pos)) * 0.45 + 0.55, width=0.01,
        color=cmap(norm(-np.array(x_pos_error))), label="Directional Error of X-Position prediction")

with open('./data/mc_context_random.json') as f:
    context = json.load(f)

context_pos = []
for p in context:
    context_pos.append(p[0])
counts, bins = np.histogram(context_pos, bins=100, range=(-1.2, 0.6))
plt.stairs(counts/800, bins, alpha=0.2, fill=True, label="Distribution of Context samples", color="tab:olive")

counts, bins = np.histogram(x_pos, bins=100, range=(-1.2, 0.6))
plt.stairs(counts/150, bins, alpha=0.2, fill=True, label="Distribution of Test samples", color="steelblue")


plt.title("Error on MountainCar-v0 relative to position with random Context")
plt.xlabel("X position of Car")
plt.ylabel("Y postion / Error / Frequency")

plt.legend()
plt.savefig("../plots/MountainCar_random_context_plot.png", dpi=500)
#plt.show()
plt.cla()
plt.clf()

# Expert Starts here

with open('./data/plotmcerror_expert.json') as f:
    data = json.load(f)
print(data.keys())

print(list(colormaps))

cmap = matplotlib.cm.get_cmap("plasma")

# plot sinodal curve
min_position = -1.2
max_position = 0.6

xs = np.linspace(min_position, max_position, 100)
ys = np.sin(3 * xs) * 0.45 + 0.55

plt.figure(figsize=(10,6))
# contour of  mountain
plt.plot(xs, ys, label="Map Contour", color="grey")

# plots error directional
x_pos = []
# is target - pred
# target is 1 pred is 0.9 -> error 0.1
x_pos_error = []
for pos, err in zip(data["state"], data["error"]):
    x_pos.append(pos[0])
    x_pos_error.append(err[0])
print(len(x_pos))
print(len(x_pos_error))
print(len(np.sin(3 * np.array(x_pos)) * 0.45 + 0.55))
norm = matplotlib.colors.Normalize(vmin=min(-np.array(x_pos_error)), vmax=max(-np.array(x_pos_error)))
plt.bar(x_pos, -np.array(x_pos_error), bottom=np.sin(3 * np.array(x_pos)) * 0.45 + 0.55, width=0.01,
        color=cmap(norm(-np.array(x_pos_error))), label="Directional Error of X-Position prediction")


with open('./data/mc_context_expert.json') as f:
    context = json.load(f)

context_pos = []
for p in context:
    context_pos.append(p[0])
counts, bins = np.histogram(context_pos, bins=100, range=(-1.2, 0.6))
plt.stairs(counts / 800, bins, alpha=0.2, fill=True, label="Distribution of Context samples", color="tab:olive")

counts, bins = np.histogram(x_pos, bins=100, range=(-1.2, 0.6))
plt.stairs(counts/150, bins, alpha=0.2, fill=True, label="Distribution of Test samples", color="steelblue")


plt.title("Error on MountainCar-v0 relative to position with Expert context")
plt.xlabel("X position of Car")
plt.ylabel("Y postion / Error / Frequency")
plt.legend()
plt.savefig("../plots/MountainCar_expert_context_plot.png", dpi=500)
#plt.show()



# Mixture Starts here

with open('./data/plotmcerror_mixture.json') as f:
    data = json.load(f)
print(data.keys())

print(list(colormaps))

cmap = matplotlib.cm.get_cmap("plasma")

# plot sinodal curve
min_position = -1.2
max_position = 0.6

xs = np.linspace(min_position, max_position, 100)
ys = np.sin(3 * xs) * 0.45 + 0.55

plt.figure(figsize=(10,6))
# contour of  mountain
plt.plot(xs, ys, label="Map Contour", color="grey")

# plots error directional
x_pos = []
# is target - pred
# target is 1 pred is 0.9 -> error 0.1
x_pos_error = []
for pos, err in zip(data["state"], data["error"]):
    x_pos.append(pos[0])
    x_pos_error.append(err[0])
print(len(x_pos))
print(len(x_pos_error))
print(len(np.sin(3 * np.array(x_pos)) * 0.45 + 0.55))
norm = matplotlib.colors.Normalize(vmin=min(-np.array(x_pos_error)), vmax=max(-np.array(x_pos_error)))
plt.bar(x_pos, -np.array(x_pos_error), bottom=np.sin(3 * np.array(x_pos)) * 0.45 + 0.55, width=0.01,
        color=cmap(norm(-np.array(x_pos_error))), label="Directional Error of X-Position prediction")


with open('./data/mc_context_mixture.json') as f:
    context = json.load(f)

context_pos = []
context = np.array(context).reshape(-1, 14)
for p in context:
    context_pos.append(p[0])
counts, bins = np.histogram(context_pos, bins=100, range=(-1.2, 0.6))
plt.stairs(counts / 800, bins, alpha=0.2, fill=True, label="Distribution of Context samples", color="tab:olive")

counts, bins = np.histogram(x_pos, bins=100, range=(-1.2, 0.6))
plt.stairs(counts/150, bins, alpha=0.2, fill=True, label="Distribution of Test samples", color="steelblue")


plt.title("Error on MountainCar-v0 relative to position with Mixture context")
plt.xlabel("X position of Car")
plt.ylabel("Y postion / Error / Frequency")
plt.legend()
plt.savefig("../plots/MountainCar_mixture_context_plot.png", dpi=500)
# plt.show()
