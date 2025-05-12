from scipy.optimize import minimize
import numpy as np


class MPCController:
    def __init__(self, predict_fn, horizon=20, dt=0.05):
        self.predict_fn = predict_fn
        self.horizon = horizon
        self.dt = dt

        # Весовые коэффициенты целевой функции
        self.w_pos = 1.0
        self.w_orient = 1.0
        self.w_u = 0.1  # на случай добавления регуляризации по управлению

        # Ограничения по thrust каждого мотора (например, в [0, 1] нормализовано)
        self.u_min = 0.0
        self.u_max = 1.0
        self.num_motors = 4

    def objective(self, u_flat, x0, target):
        u_seq = u_flat.reshape((self.horizon, self.num_motors))
        x = x0.copy()
        total_cost = 0.0

        for i in range(self.horizon):
            u = u_seq[i]
            x = self.predict_fn(x, u, self.dt)
            pos_error = np.linalg.norm(x[0:3] - target[0:3])  # x, y, z
            orient_error = np.linalg.norm(x[6:9] - target[6:9])  # roll, pitch, yaw
            cost = self.w_pos * pos_error**2 + self.w_orient * orient_error**2
            total_cost += cost

        return total_cost

    def control(self, x0, target):
        u0 = np.ones((self.horizon, self.num_motors)) * 0.5
        bounds = [(self.u_min, self.u_max)] * self.horizon * self.num_motors

        res = minimize(
            self.objective,
            u0.flatten(),
            args=(x0, target),
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 50, 'disp': False}
        )

        if res.success:
            u_opt = res.x.reshape((self.horizon, self.num_motors))
            return u_opt[0]  # Только первое управляющее воздействие
        else:
            print("[MPC] Optimization failed")
            return np.ones(self.num_motors) * 0.5