"""Print diffusers DPM-Solver++(2M) reference values for src/scheduler.rs tests.

    python tools/dump_dpm_reference.py
"""
import torch
from diffusers import DPMSolverMultistepScheduler

s = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                num_train_timesteps=1000, algorithm_type="dpmsolver++", solver_order=2,
                                lower_order_final=True, timestep_spacing="trailing",
                                final_sigmas_type="zero", prediction_type="epsilon")
s.set_timesteps(5)
print("timesteps", s.timesteps.tolist())
x = torch.tensor([0.5, -1.25, 2.0], dtype=torch.float64)
for i, t in enumerate(s.timesteps):
    eps = torch.tensor([0.1, -0.3, 0.7], dtype=torch.float64) * (1.0 + 0.2 * i)
    x = s.step(eps, t, x).prev_sample
    print(f"after step {i} (t={int(t)}):", [f"{v:.9e}" for v in x.tolist()])
