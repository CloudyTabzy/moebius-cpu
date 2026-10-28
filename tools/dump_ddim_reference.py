"""Print diffusers DDIM reference values used by src/scheduler.rs tests.

Matches the scheduler built in Moebius infer/utils.py::build_pipeline and the
timestep slicing in removal/v1_2/pipeline.py::get_timesteps.

    python tools/dump_ddim_reference.py
"""
import torch
from diffusers import DDIMScheduler

s = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                  num_train_timesteps=1000, clip_sample=False)
for steps, strength in [(20, 0.99), (7, 0.99), (20, 1.0)]:
    s.set_timesteps(steps)
    init = min(int(steps * strength), steps)
    t_start = max(steps - init, 0)
    print(f"steps={steps} strength={strength}: {s.timesteps[t_start * s.order:].tolist()}")

ac = s.alphas_cumprod
print("alphas_cumprod[0,1,500,900,999] =", [f"{ac[i].item():.9e}" for i in (0, 1, 500, 900, 999)])

s.set_timesteps(20)
x = torch.tensor([0.5, -1.25, 2.0])
eps = torch.tensor([0.1, -0.3, 0.7])
for t in (900, 50, 0):
    out = s.step(eps, t, x).prev_sample
    print(f"step(t={t}) =", [f"{v:.9e}" for v in out.tolist()])
noisy = s.add_noise(x, eps, torch.tensor([900]))
print("add_noise(t=900) =", [f"{v:.9e}" for v in noisy.tolist()])
