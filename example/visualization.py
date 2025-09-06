import numpy as np
import matplotlib.pyplot as plt

TRAJECTORIES_TO_VISUALIZE = 100


# Basic step function -- apply the control to advance one step
def euler(vehicle, state, control):
    return vehicle.ode(state, control)


#
# Also define the Runge-Kutta variant as it is (apparently) a much
# better approximation of the first order derivative
#
# https://en.wikipedia.org/wiki/Runge-Kutta_methods
def runge_kutta_step(vehicle, state, control, dt):
    k1 = vehicle.ode(state, control)
    k2 = vehicle.ode(state + k1 * (dt / 2), control)
    k3 = vehicle.ode(state + k2 * (dt / 2), control)
    k4 = vehicle.ode(state + k3 * dt, control)

    return (k1 + 2 * (k2 + k3) + k4) / 6.0


# wrapper to allow easy switch between methods. (simplifies validation)
def step_fn(vehicle, state, control, dt=None):
    # return euler(vehicle=vehicle, state=state, control=control)
    return runge_kutta_step(vehicle=vehicle, state=state, control=control, dt=dt)


def run_trajectory(vehicle, initial_state, controls, dt):

    traj = np.zeros((len(controls) + 1, len(initial_state)))
    traj[0, :] = initial_state

    state = np.array(initial_state)
    for m, u in enumerate(controls):
        step = step_fn(vehicle=vehicle, state=state, control=u, dt=dt)
        state += step * dt
        traj[m + 1, :] = state

    return traj


def visualize_variations(
    figure,
    vehicle,
    initial_state,
    u_nom,
    u_variations,
    u_weighted,
    dt,
    figure_name=None,
):
    # visualizing!
    fig, ax = plt.subplots(num=figure, nrows=1, ncols=3, figsize=(15, 5))

    n_samples, n_controls, n_steps = u_variations.shape
    indexes = np.random.choice(
        n_samples, min(n_samples, TRAJECTORIES_TO_VISUALIZE), replace=False
    )

    new_traj_pts = []
    for i in indexes:
        u_var = np.array(u_nom)
        u_var = u_var + u_variations[i, ...].T

        traj = run_trajectory(
            vehicle=vehicle, initial_state=initial_state, controls=u_var, dt=dt
        )
        new_traj_pts.append(np.expand_dims(traj, axis=0))

    new_traj_pts = np.vstack(new_traj_pts)

    traj = run_trajectory(
        vehicle=vehicle, initial_state=initial_state, controls=u_weighted, dt=dt
    )

    nom_traj = run_trajectory(
        vehicle=vehicle, initial_state=initial_state, controls=u_nom, dt=dt
    )

    ax[0].plot(nom_traj[:, 0], nom_traj[:, 1])
    ax[1].plot(new_traj_pts[:, :, 0].T, new_traj_pts[:, :, 1].T)
    ax[2].plot(traj[:, 0], traj[:, 1])

    ax[0].axis("equal")
    ax[1].axis("equal")
    ax[2].axis("equal")

    ax[0].set_title("Nominal trajectory")
    ax[1].set_title("Sampled trajectories")
    ax[2].set_title("Weighted trajectory")

    if figure_name is not None:
        fig.savefig(figure_name)

    return fig, ax
