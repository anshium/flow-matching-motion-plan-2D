from imports_self import *
from config import *

from dataset.path_dataset import PathDataset, OBSTACLE_MAP_SIZE

from main_modules import ObstacleEncoder, UNet1DPathFlow

# --- Main Training Script ---
# 1. Data
train_dataset = PathDataset(NUM_TRAINING_SAMPLES, OBSTACLE_MAP_SIZE, N_WAYPOINTS, PATH_DIM) # More samples for real
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 2. Models
obstacle_enc = ObstacleEncoder(
    map_shape=OBSTACLE_MAP_SIZE,
    global_embed_dim=OBSTACLE_ENCODER_EMBED_DIM,
    context_tokens=OBSTACLE_CONTEXT_TOKENS if USE_CROSS_ATTN_IN_UNET else None,
    context_dim_per_token=OBSTACLE_CONTEXT_DIM if USE_CROSS_ATTN_IN_UNET else None
).to(DEVICE)

# Determine self-attention and cross-attention levels (e.g., deeper layers)
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

# 3. Optimizer
optimizer = optim.Adam(
    list(obstacle_enc.parameters()) + list(path_flow_unet.parameters()),
    lr=LEARNING_RATE
)

# 4. CFM
cfm = ConditionalFlowMatcher(sigma=CFM_SIGMA)
obstacle_enc
# 5. Training Loop
print("Starting training...")
for epoch in range(N_EPOCHS):
    path_flow_unet.train()
    obstacle_enc.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
    for batch_idx, (obs_maps, starts, goals, expert_paths) in enumerate(progress_bar):
        optimizer.zero_grad()

        obs_maps = obs_maps.to(DEVICE)
        starts = starts.to(DEVICE)
        goals = goals.to(DEVICE)
        expert_paths = expert_paths.to(DEVICE) # x1

        # Get obstacle embeddings
        # y_obs_context will be None if context_tokens not set in ObstacleEncoder
        y_obs_global, y_obs_context = obstacle_enc(obs_maps)

        # Create global conditioning vector
        y_condition_global = torch.cat([starts, goals, y_obs_global], dim=1)

        # CFM sampling
        x0_paths = torch.randn_like(expert_paths)
        t, xt_paths, ut_paths = cfm.sample_location_and_conditional_flow(x0_paths, expert_paths)
        t = t.to(DEVICE)

        # Model prediction
        # Pass y_obs_context only if USE_CROSS_ATTN_IN_UNET is True and it's available
        effective_y_obs_context = y_obs_context if USE_CROSS_ATTN_IN_UNET and y_obs_context is not None else None
        vt_paths = path_flow_unet(t, xt_paths, y_condition_global, effective_y_obs_context)

        loss = F.mse_loss(vt_paths, ut_paths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{N_EPOCHS}, Average Loss: {avg_loss:.4f}")

print("Training finished.")


torch.save(obstacle_enc.state_dict(), "obstacle_encoder.pth")
torch.save(path_flow_unet.state_dict(), "path_flow_unet.pth")

torch.save(optimizer.state_dict(), "optimizer.pth")

torch.save({
    "epoch": N_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "cfm_sigma": CFM_SIGMA
}, "training_config.pth")

torch.save({
    "obstacle_maps": train_dataset.obstacle_maps,
    "start_positions": train_dataset.start_pos,
    "goal_positions": train_dataset.goal_pos,
    "expert_paths": train_dataset.expert_paths
}, "path_dataset.pth")

torch.save({
    "epoch": N_EPOCHS,
    "total_loss": avg_loss
}, "training_state.pth")