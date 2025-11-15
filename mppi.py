import numpy as np
import mujoco
import mujoco.rollout
import mujoco.viewer
class MPPI:
    def __init__(
        self,
        model,
        data,
        horizon,
        num_samples,
        lambda_,
        noise_sigma,
        dt,
        cost_fn,
        frame_skip=1,
        u_min=None,
        u_max=None,
        num_threads = 32
    ):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_
        self.noise_sigma = noise_sigma
        self.dt = dt
        self.cost_fn = cost_fn
        self.frame_skip = frame_skip
        self.num_threads = num_threads

        # allocate data for rollouts
        self.data_pool = [mujoco.MjData(model) for _ in range(num_threads)]

        self.n_u = model.nu
        self.internal_horizon = horizon * frame_skip
        self.u_min = np.full(self.n_u, -np.inf) if u_min is None else u_min
        self.u_max = np.full(self.n_u,  np.inf) if u_max is None else u_max
        self.u = np.zeros((horizon, self.n_u))

    def _expand_controls(self, ctrl):
        # ctrl: (N, H, nu)   →   expanded: (N, H*frame_skip, nu)
        return np.repeat(ctrl, self.frame_skip, axis=1)

    def command(self, initial_state):
        noise = np.random.randn(self.num_samples, self.horizon, self.n_u) * self.noise_sigma
        controls = np.clip(self.u[None, :, :] + noise, self.u_min, self.u_max)

        expanded_ctrl = self._expand_controls(controls)

        print("States shape:", initial_state.shape)  # Debugging line
        print("Expanded controls shape:", expanded_ctrl.shape)  # Debugging line

        states, _ = mujoco.rollout.rollout(
            self.model,
            self.data_pool,
            np.repeat(initial_state[None, :], self.num_samples, axis=0),
            control=expanded_ctrl,
            nstep=self.internal_horizon,
        )

        
        costs = self.cost_fn(states, controls)  # cost on MPPI-level horizon

        beta = np.min(costs)
        weights = np.exp(-(costs - beta) / self.lambda_)
        weights /= np.sum(weights)

        # print shapes for debugging
        # print("Costs shape:", costs.shape)  # Debugging line
        print("costs"   , costs)  # Debugging line
        print("Weights", weights)  # Debugging line
        # print("Noise shape:", noise.shape)  # Debugging line

        # du = np.einsum('k,kij->ij', weights, noise)
        # weighted sum
        du = weights[:, None, None] * noise
        du = np.sum(du, axis=0)
        self.u += du

        action = self.u[0]
        self.u[:-1] = self.u[1:]
        self.u[-1] = 0

        return np.clip(action, self.u_min, self.u_max)

def walking_reward_gt(state, control): #input:(x,u)
    upright = (np.cos(state[...,3]) + 1)/2
    #standing = np.exp(-16*(state[...,0])**2)
    standing = np.clip(1-1.0*np.abs(state[...,2]),0,1)
    standing_reward = (3*standing + upright)/4

    #move_reward = np.exp(-8*(1.0-state[...,8])**2)
    move_reward = np.clip(state[...,10]/1.0,0.0,1)
    return standing_reward * move_reward

def walking_cost_fn(states, controls):
    rewards = walking_reward_gt(states, controls) # shape: (N, H)
    total_cost = -np.mean(rewards, axis=1)
    return total_cost


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("assets/walker/muscle_new.xml")
    data = mujoco.MjData(model)

    # viewer
    viewer = mujoco.viewer.launch_passive(model, data)

    horizon = 40
    num_samples = 512
    lambda_ = 0.01
    noise_sigma = 1.0
    dt = model.opt.timestep
    frame_skip = 10

    mppi_controller = MPPI(
        model=model,
        data=data,
        horizon=horizon,
        num_samples=num_samples,
        lambda_=lambda_,
        noise_sigma=noise_sigma,
        dt=dt,
        cost_fn=walking_cost_fn,
        frame_skip=frame_skip,
        num_threads=32,
        u_max=np.array([1.0]*model.nu),
        u_min=np.array([0.0]*model.nu),
    )

    for i in range(1000):
        full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
        initial_state = np.zeros((mujoco.mj_stateSize(model, full_physics),))
        mujoco.mj_getState(model, data, initial_state, full_physics)
        print("Initial state:", initial_state)  # Debugging line
        action = mppi_controller.command(initial_state)

        data.ctrl[:] = action

        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
            viewer.sync()
    
