from time import time
from car_model import Ackermann4
import numpy as np
import matplotlib.pyplot as plt

from warp_mppi import MPPI
from visualization import visualize_variations

samples_demo = 1
seed_demo = int(time())  # More varied seed
u_dist_limits_demo = [2, np.pi / 4]
u_limits_demo = [4, np.pi / 5]
M_demo = 0.001
Q_demo = [1, 1, 1, 1]
R_demo = [1.0, 1.0]
method_demo = "Ours"
c_lambda_demo = 100
Qf_demo = [10, 10, 1, 10]  # Example final state cost

vehicle_demo = Ackermann4()

tic = time()
mppi = MPPI(
    vehicle=vehicle_demo,
    samples=samples_demo,
    seed=seed_demo,
    u_limits=u_limits_demo,
    u_dist_limits=u_dist_limits_demo,
    M=M_demo,
    Q=Q_demo,
    Qf=Qf_demo,
    R=R_demo,
    method=method_demo,
    c_lambda=c_lambda_demo,
    scan_range=30,
    vehicle_length=vehicle_demo.L,
)

costmap = np.ones((100, 100)) * 0.5
origin = (0, 0)
resolution = 1
x_nom = np.zeros((10, 4))
for i in range(10):  # Create a simple nominal trajectory
    x_nom[i, 0] = i * 0.5  # x
    x_nom[i, 1] = 0  # y
    x_nom[i, 2] = 0.5  # v
    x_nom[i, 3] = 0  # theta

u_nom = np.ones((9, 2))
u_nom[:, 0] = 0.1  # acceleration
u_nom[:, 1] = 0.0  # steering

x_init = np.array([0.0, 0.0, 1.0, 0.0])
x_goal = np.array(
    [x_nom[-1, 0], x_nom[-1, 1], x_nom[-1, 2], x_nom[-1, 3]]
)  # Goal is the end of nominal trajectory

actors = [
    [9, 0, 5, 13, 5, np.sqrt(15 * 15 + 5 * 5)],
    [20, 20, 5, 25, 5, np.sqrt(25 * 25 + 5 * 5)],
]

dt = 0.1
u_mppi, u_dist, weights = mppi.find_control(
    costmap, origin, resolution, x_init, x_goal, x_nom, u_nom, actors, dt
)

toc = time()
print(f"Time: {toc - tic}, per sample: {(toc - tic) / samples_demo}")

fig, ax = visualize_variations(
    figure=1,
    vehicle=Ackermann4(),
    initial_state=x_init,
    u_nom=u_nom,
    u_variations=u_dist,
    u_weighted=u_mppi,
    weights=weights,
    dt=dt,
)

plt.show(block=False)

print(u_mppi)
print(u_dist)
