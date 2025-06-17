from imports_self import *
from config import *

from dataset.path_dataset import PathDataset, OBSTACLE_MAP_SIZE

from main_modules import ObstacleEncoder, UNet1DPathFlow

obstacle_enc = ObstacleEncoder(
    map_shape=OBSTACLE_MAP_SIZE,
    global_embed_dim=OBSTACLE_ENCODER_EMBED_DIM,
    context_tokens=OBSTACLE_CONTEXT_TOKENS if USE_CROSS_ATTN_IN_UNET else None,
    context_dim_per_token=OBSTACLE_CONTEXT_DIM if USE_CROSS_ATTN_IN_UNET else None
).to(DEVICE)

unet_levels = len(UNET_CHANNEL_MULTS)
self_attn_lvls = tuple(range(unet_levels // 2, unet_levels + 1)) if USE_SELF_ATTN_IN_UNET else () # Attn in bottleneck too
cross_attn_lvls = tuple(range(unet_levels // 2, unet_levels)) if USE_CROSS_ATTN_IN_UNET else ()

path_flow_unet = UNet1DPathFlow(
    path_dim=PATH_DIM,
    init_ch=UNET_INIT_CONV_CHANNELS,
    ch_mults=UNET_CHANNEL_MULTS,
    t_emb_dim=UNET_TIME_EMB_DIM,
    y_glob_dim=UNET_GLOBAL_COND_DIM,
    y_cross_dim=OBSTACLE_CONTEXT_DIM if USE_CROSS_ATTN_IN_UNET else None,
    n_conv_layers=UNET_NUM_CONV_LAYERS,
    self_attn_levels=self_attn_lvls,
    cross_attn_levels=cross_attn_lvls,
    attn_heads=UNET_ATTN_HEADS,
    attn_dim_head=UNET_ATTN_DIM_HEAD
).to(DEVICE)

obstacle_enc.load_state_dict(torch.load("obstacle_encoder.pth"))
path_flow_unet.load_state_dict(torch.load("path_flow_unet.pth"))
optimizer = optim.Adam(
    list(obstacle_enc.parameters()) + list(path_flow_unet.parameters()),
    lr=LEARNING_RATE
)
optimizer.load_state_dict(torch.load("optimizer.pth"))

training_config = torch.load("training_config.pth")

train_dataset = PathDataset(1024, OBSTACLE_MAP_SIZE, N_WAYPOINTS, PATH_DIM)

training_state = torch.load("training_state.pth")

avg_loss = training_state["total_loss"]

# 6. Inference/Sampling Example
print("Generating a sample path...")
path_flow_unet.eval()
obstacle_enc.eval()

# sample from the dataset for conditioning
sample_obs_map, sample_start_norm, sample_goal_norm, _ = train_dataset[0]
sample_obs_map = sample_obs_map.unsqueeze(0).to(DEVICE)
sample_start_norm = sample_start_norm.unsqueeze(0).to(DEVICE)
sample_goal_norm = sample_goal_norm.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    y_obs_global_sample, y_obs_context_sample = obstacle_enc(sample_obs_map)
    y_cond_global_sample = torch.cat([sample_start_norm, sample_goal_norm, y_obs_global_sample], dim=1)
    effective_y_obs_context_sample = y_obs_context_sample if USE_CROSS_ATTN_IN_UNET and y_obs_context_sample is not None else None


    # Initial noise path for generation
    initial_noise = torch.randn(1, N_WAYPOINTS, PATH_DIM, device=DEVICE)

    # ODE function for the solver
    def ode_func(t_scalar, current_path_flat):
        current_path = current_path_flat.view(1, N_WAYPOINTS, PATH_DIM)
        # Solver gives scalar t, model might expect (B,) or (B,1)
        t_tensor = torch.tensor([t_scalar], device=DEVICE).float()
        # Conditioning vectors are already [1, D_cond]
        vt = path_flow_unet(t_tensor, current_path, y_cond_global_sample, effective_y_obs_context_sample)
        return vt.flatten() # Solver expects flattened output

    # Time steps for generation
    t_span = torch.linspace(0, 1, 100, device=DEVICE) # More steps for smoother generation

    # solving the ODE
    generated_path_flat_traj = torchdiffeq.odeint(
        ode_func,
        initial_noise.flatten(),
        t_span,
        method='dopri5',
        atol=1e-4,
        rtol=1e-4
    )
    generated_path_flat = generated_path_flat_traj[-1] # Get the path at t=1
    generated_path_norm = generated_path_flat.view(1, N_WAYPOINTS, PATH_DIM)

# denormalize the generated path and sample start/goal
map_max_dim = max(OBSTACLE_MAP_SIZE)
generated_path_denorm = (generated_path_norm.squeeze(0).cpu().numpy() + 1) / 2 * map_max_dim
sample_start_denorm = (sample_start_norm.squeeze(0).cpu().numpy() + 1) / 2 * map_max_dim
sample_goal_denorm = (sample_goal_norm.squeeze(0).cpu().numpy() + 1) / 2 * map_max_dim

print("Generated path shape:", generated_path_denorm.shape)

if isinstance(sample_obs_map, torch.Tensor):
    obs_map_np = sample_obs_map.cpu().numpy()
else:
    obs_map_np = np.array(sample_obs_map)

while obs_map_np.ndim > 2 and obs_map_np.shape[0] == 1:
    obs_map_np = obs_map_np.squeeze(0)
if obs_map_np.ndim == 3 and obs_map_np.shape[0] == 1:
    obs_map_np = obs_map_np.squeeze(0)
elif obs_map_np.ndim != 2:
    raise ValueError(f"Obstacle map has unexpected dimensions: {obs_map_np.shape}. Expected 2D (H,W).")


if isinstance(generated_path_denorm, torch.Tensor):
    path_np = generated_path_denorm.cpu().numpy()
else:
    path_np = np.array(generated_path_denorm)

if isinstance(sample_start_denorm, torch.Tensor):
    start_np = sample_start_denorm.cpu().numpy()
else:
    start_np = np.array(sample_start_denorm)

if isinstance(sample_goal_denorm, torch.Tensor):
    goal_np = sample_goal_denorm.cpu().numpy()
else:
    goal_np = np.array(sample_goal_denorm)

output_filename = "visualization_sample.npz"

np.savez(
    output_filename,
    obstacle_map=obs_map_np,
    generated_path=path_np,
    start_position=start_np,
    goal_position=goal_np,
    map_size=np.array(OBSTACLE_MAP_SIZE)
)
print(f"Visualization data saved to {output_filename}")


# 7. Visualization
# plt.figure(figsize=(6,6))
# plt.imshow(sample_obs_map.squeeze(0).squeeze(0).cpu().numpy().T, cmap='gray_r', origin='lower', extent=[0, OBSTACLE_MAP_SIZE[0], 0, OBSTACLE_MAP_SIZE[1]])
# plt.plot(generated_path_denorm[:, 0], generated_path_denorm[:, 1], 'b-o', label='Generated Path', markersize=3)
# plt.plot(sample_start_denorm[0], sample_start_denorm[1], 'go', label='Start', markersize=8)
# plt.plot(sample_goal_denorm[0], sample_goal_denorm[1], 'ro', label='Goal', markersize=8)
# plt.title("Generated Path with Obstacles")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.grid(True)
# plt.xlim(0, OBSTACLE_MAP_SIZE[0])
# plt.ylim(0, OBSTACLE_MAP_SIZE[1])
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()