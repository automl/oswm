import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import colormaps


with open('./data/plotcartpoleerror.json') as f:
    data = json.load(f)

cmap = matplotlib.cm.get_cmap("plasma")
plt.figure(figsize=(10,6))

# plots error directional
rad_pos = []
ang_vel = []
# is target - pred
# negative rotation means right hand side
rad_pos_error = []
for pos, err in zip(data["state"], data["error"]):
    rad_pos.append(np.arcsin(pos[2]))
    ang_vel.append(pos[3])
    rad_pos_error.append(rad_pos[-1] - np.arcsin(pos[2] + err[2]))

ang_vel = np.array(ang_vel)

rad_pos = np.array(rad_pos) + np.pi * 0.5
rad_pos = np.where(rad_pos > np.pi, -2 * np.pi + rad_pos, rad_pos)

#rad_pos_error = np.array(rad_pos_error)
#rad_pos_error = np.where(rad_pos_error > 1, 0, rad_pos_error)
#rad_pos_error = np.where(rad_pos_error < -1, 0, rad_pos_error)


norm = matplotlib.colors.Normalize(vmin=min(-np.array(rad_pos_error)), vmax=max(-np.array(rad_pos_error)))
plt.bar(rad_pos, -np.array(rad_pos_error), bottom=.5, width=0.002, color=cmap(norm(-np.array(rad_pos_error))), label="Directional Error of X-Position prediction")

with open('./data/cartpole_context_random.json') as f:
    context = json.load(f)

context = np.array(context).reshape(-1, 14)
context_pos = []
context_vel = []
for p in context:
    context_pos.append(np.arcsin(p[2]))
    context_vel.append(p[3])
context_vel = np.array(context_vel)

context_pos = np.array(context_pos) + np.pi * 0.5
context_pos = np.where(context_pos > np.pi, -2 * np.pi + context_pos, context_pos)
counts, bins = np.histogram(context_pos, bins=100, range=(0.5 * np.pi - 0.24, 0.5 * np.pi + 0.24))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Context samples", color="tab:olive")

counts, bins = np.histogram(rad_pos, bins=100, range=(0.5 * np.pi - 0.24, 0.5 * np.pi + 0.24))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Test samples", color="steelblue")


plt.title("Error on CartPole-v0 relative to position with random Context")
plt.xlabel("Angle of Pendulum")
plt.ylabel("Error / Frequency")

plt.legend()
plt.ylim([0, 1.0])
plt.savefig("../plots/Cartpole_random_context_plot.png", dpi=500)
plt.figure(figsize=(10,6))

plt.cla()
plt.clf()

plt.title("Error on CartPole-v0 relative to velocity with random Context")
plt.bar(ang_vel, -np.array(rad_pos_error), bottom=.5, width=0.025, color=cmap(norm(-np.array(rad_pos_error))), label="Directional Error of X-Position prediction")
counts, bins = np.histogram(context_vel, bins=100, range=(context_vel.min(), context_vel.max()))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Context samples", color="tab:olive")

counts, bins = np.histogram(ang_vel, bins=100, range=(ang_vel.min(), ang_vel.max()))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Test samples", color="steelblue")
plt.ylabel("Error")
plt.xlabel("Angular velocity")
plt.legend()
plt.ylim([0, 1.0])
plt.savefig("../plots/Cartpole_vel_random_context_plot.png", dpi=500)

plt.cla()
plt.clf()


# EXPERT CONTEXT STARTS HERE
with open('./data/plotcartpoleerror_expert.json') as f:
    data = json.load(f)

cmap = matplotlib.cm.get_cmap("plasma")
plt.figure(figsize=(10,6))

# plots error directional
rad_pos = []
ang_vel = []
# is target - pred
# negative rotation means right hand side
rad_pos_error = []
for pos, err in zip(data["state"], data["error"]):
    rad_pos.append(np.arcsin(pos[2]))
    ang_vel.append(pos[3])
    rad_pos_error.append(rad_pos[-1] - np.arcsin(pos[2] + err[2]))

ang_vel = np.array(ang_vel)

rad_pos = np.array(rad_pos) + np.pi * 0.5
rad_pos = np.where(rad_pos > np.pi, -2 * np.pi + rad_pos, rad_pos)

#rad_pos_error = np.array(rad_pos_error)
#rad_pos_error = np.where(rad_pos_error > 1, 0, rad_pos_error)
#rad_pos_error = np.where(rad_pos_error < -1, 0, rad_pos_error)


norm = matplotlib.colors.Normalize(vmin=min(-np.array(rad_pos_error)), vmax=max(-np.array(rad_pos_error)))
plt.bar(rad_pos, -np.array(rad_pos_error), bottom=.5, width=0.002, color=cmap(norm(-np.array(rad_pos_error))), label="Directional Error of X-Position prediction")

with open('./data/cartpole_context_expert.json') as f:
    context = json.load(f)

context = np.array(context).reshape(-1, 14)
context_pos = []
context_vel = []
for p in context:
    context_pos.append(np.arcsin(p[2]))
    context_vel.append(p[3])
context_vel = np.array(context_vel)

context_pos = np.array(context_pos) + np.pi * 0.5
context_pos = np.where(context_pos > np.pi, -2 * np.pi + context_pos, context_pos)
counts, bins = np.histogram(context_pos, bins=100, range=(0.5 * np.pi - 0.24, 0.5 * np.pi + 0.24))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Context samples", color="tab:olive")

counts, bins = np.histogram(rad_pos, bins=100, range=(0.5 * np.pi - 0.24, 0.5 * np.pi + 0.24))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Test samples", color="steelblue")


plt.title("Error on CartPole-v0 relative to position with Expert Context")
plt.xlabel("Angle of Pendulum")
plt.ylabel("Error / Frequency")

plt.legend()
# plt.savefig("MountainCar_random_context_plot.png", dpi=500)
plt.ylim([0, 1.0])
plt.savefig("../plots/Cartpole_expert_context_plot.png", dpi=500)
plt.figure(figsize=(10,6))

#plt.cla()
#plt.clf()

plt.title("Error on CartPole-v0 relative to velocity with Expert Context")
plt.bar(ang_vel, -np.array(rad_pos_error), bottom=.5, width=0.025, color=cmap(norm(-np.array(rad_pos_error))), label="Directional Error of X-Position prediction")
counts, bins = np.histogram(context_vel, bins=100, range=(context_vel.min(), context_vel.max()))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Context samples", color="tab:olive")

counts, bins = np.histogram(ang_vel, bins=100, range=(ang_vel.min(), ang_vel.max()))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Test samples", color="steelblue")
plt.ylabel("Error")
plt.xlabel("Angular velocity")
plt.legend()
plt.ylim([0, 1.0])
plt.savefig("../plots/Cartpole_vel_expert_context_plot.png", dpi=500)


# MIXTURe CONTEXT STARTS HERE
with open('plotcartpoleerror_mixture.json') as f:
    data = json.load(f)

cmap = matplotlib.cm.get_cmap("plasma")
plt.figure(figsize=(10,6))

# plots error directional
rad_pos = []
ang_vel = []
# is target - pred
# negative rotation means right hand side
rad_pos_error = []
for pos, err in zip(data["state"], data["error"]):
    rad_pos.append(np.arcsin(pos[2]))
    ang_vel.append(pos[3])
    rad_pos_error.append(rad_pos[-1] - np.arcsin(pos[2] + err[2]))

ang_vel = np.array(ang_vel)

rad_pos = np.array(rad_pos) + np.pi * 0.5
rad_pos = np.where(rad_pos > np.pi, -2 * np.pi + rad_pos, rad_pos)

#rad_pos_error = np.array(rad_pos_error)
#rad_pos_error = np.where(rad_pos_error > 1, 0, rad_pos_error)
#rad_pos_error = np.where(rad_pos_error < -1, 0, rad_pos_error)


norm = matplotlib.colors.Normalize(vmin=min(-np.array(rad_pos_error)), vmax=max(-np.array(rad_pos_error)))
plt.bar(rad_pos, -np.array(rad_pos_error), bottom=.5, width=0.002, color=cmap(norm(-np.array(rad_pos_error))), label="Directional Error of X-Position prediction")

with open('./data/cartpole_context_mixture.json') as f:
    context = json.load(f)

context = np.array(context).reshape(-1, 14)
context_pos = []
context_vel = []
for p in context:
    context_pos.append(np.arcsin(p[2]))
    context_vel.append(p[3])
context_vel = np.array(context_vel)

context_pos = np.array(context_pos) + np.pi * 0.5
context_pos = np.where(context_pos > np.pi, -2 * np.pi + context_pos, context_pos)
counts, bins = np.histogram(context_pos, bins=100, range=(0.5 * np.pi - 0.24, 0.5 * np.pi + 0.24))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Context samples", color="tab:olive")

counts, bins = np.histogram(rad_pos, bins=100, range=(0.5 * np.pi - 0.24, 0.5 * np.pi + 0.24))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Test samples", color="steelblue")


plt.title("Error on CartPole-v0 relative to position with Mixture Context")
plt.xlabel("Angle of Pendulum")
plt.ylabel("Error / Frequency")

plt.legend()
# plt.savefig("MountainCar_random_context_plot.png", dpi=500)
plt.ylim([0, 1.0])
plt.savefig("../plots/Cartpole_mixture_context_plot.png", dpi=500)
plt.figure(figsize=(10,6))
#plt.cla()
#plt.clf()

plt.title("Error on CartPole-v0 relative to velocity with Mixture Context")
plt.bar(ang_vel, -np.array(rad_pos_error), bottom=.5, width=0.025, color=cmap(norm(-np.array(rad_pos_error))), label="Directional Error of X-Position prediction")
counts, bins = np.histogram(context_vel, bins=100, range=(context_vel.min(), context_vel.max()))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Context samples", color="tab:olive")

counts, bins = np.histogram(ang_vel, bins=100, range=(ang_vel.min(), ang_vel.max()))
plt.stairs(counts/1000, bins, alpha=.3, fill=True, label="Distribution of Test samples", color="steelblue")
plt.ylabel("Error")
plt.xlabel("Angular velocity")
plt.legend()
plt.ylim([0, 1.0])
plt.savefig("../plots/Cartpole_vel_mixture_context_plot.png", dpi=500)
