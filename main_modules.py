from imports_self import *
from helper_modules import SinusoidalTimeEmbedding, FiLMLayer, SimpleAttention
from _1D_UNet_comps import ConvBlock, DownBlock, UpBlock

# --- Main Models: ObstacleEncoder and UNet1DPathFlow ---
class ObstacleEncoder(nn.Module):
    def __init__(self, map_shape, global_embed_dim, context_tokens=None, context_dim_per_token=None):
        super().__init__()
        self.map_shape = map_shape
        self.global_embed_dim = global_embed_dim
        self.context_tokens = context_tokens
        self.context_dim_per_token = context_dim_per_token

        # Simple CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # map_shape / 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # map_shape / 4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # map_shape / 8
        )
        # Calculate flattened size after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *map_shape)
            cnn_out_shape = self.cnn(dummy_input).shape
            self.flattened_size = cnn_out_shape[1] * cnn_out_shape[2] * cnn_out_shape[3]
            self.cnn_out_channels = cnn_out_shape[1] # Channels before flattening

        self.fc_global = nn.Linear(self.flattened_size, global_embed_dim)

        if context_tokens and context_dim_per_token:
            # For context, project from channel dim to (tokens * dim_per_token)
            # then reshape. Alternative: Use different CNN head or attention pooling.
            self.fc_context = nn.Linear(self.flattened_size, context_tokens * context_dim_per_token)
        else:
            self.fc_context = None

    def forward(self, obstacle_map): # obstacle_map: (B, 1, H, W)
        x = self.cnn(obstacle_map)
        x_flat = x.view(x.size(0), -1)
        global_embedding = self.fc_global(x_flat) # (B, global_embed_dim)

        context_embeddings = None
        if self.fc_context:
            context_flat = self.fc_context(x_flat) # (B, context_tokens * context_dim_per_token)
            context_embeddings = context_flat.view(x.size(0), self.context_tokens, self.context_dim_per_token)
        return global_embedding, context_embeddings

class UNet1DPathFlow(nn.Module):
    def __init__(self, path_dim, init_ch, ch_mults, t_emb_dim, y_glob_dim,
                 y_cross_dim=None, n_conv_layers=2, self_attn_levels=(), cross_attn_levels=(),
                 attn_heads=4, attn_dim_head=32):
        super().__init__()
        self.path_dim = path_dim
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(t_emb_dim),
                                      nn.Linear(t_emb_dim, t_emb_dim), nn.SiLU(),
                                      nn.Linear(t_emb_dim, t_emb_dim))
        self.init_conv = nn.Conv1d(path_dim, init_ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current_ch = init_ch
        num_levels = len(ch_mults)

        for i in range(num_levels):
            out_ch = init_ch * ch_mults[i]
            use_sa = i in self_attn_levels
            self.down_blocks.append(DownBlock(current_ch, out_ch, t_emb_dim, y_glob_dim, n_conv_layers, use_sa, attn_heads, attn_dim_head))
            current_ch = out_ch

        self.bottleneck = nn.ModuleList([
            ConvBlock(current_ch, current_ch*2),
            SimpleAttention(current_ch*2, heads=attn_heads, dim_head=attn_dim_head) if num_levels in self_attn_levels else nn.Identity(),
            FiLMLayer(current_ch*2, t_emb_dim),
            FiLMLayer(current_ch*2, y_glob_dim),
            ConvBlock(current_ch*2, current_ch)
        ])


        for i in reversed(range(num_levels)):
            out_ch = init_ch * ch_mults[i]
            use_ca = i in cross_attn_levels
            # UpBlock input channels: current_ch (from deeper layer) + out_ch (from skip)
            # in_ch for UpBlock is actually current_ch from previous level (after bottleneck or deeper UpBlock)
            # out_ch for UpBlock is the target channel dim for this level.
            # Upsample halves channels, so ConvBlock input is (current_ch//2 + skip_out_ch)
            self.up_blocks.append(UpBlock(current_ch, out_ch, t_emb_dim, y_glob_dim, y_cross_dim,
                                          n_conv_layers, use_ca, attn_heads, attn_dim_head))
            current_ch = out_ch

        self.final_conv = nn.Conv1d(init_ch, path_dim, 1)

    def forward(self, t, xt_path, y_glob, y_cross=None): # xt_path: (B, N, D_path)
        xt_path = xt_path.permute(0, 2, 1) # (B, D_path, N)
        t_emb = self.time_mlp(t)
        h = self.init_conv(xt_path)
        skips = [h]

        for block in self.down_blocks:
            h, skip = block(h, t_emb, y_glob)
            skips.append(skip)

        h = self.bottleneck[0](h) # Conv1
        if not isinstance(self.bottleneck[1], nn.Identity): # Attention
             h_attn = h.permute(0,2,1); h_attn = self.bottleneck[1](h_attn); h = h + h_attn.permute(0,2,1)
        h = self.bottleneck[2](h, t_emb) # FiLM time
        h = self.bottleneck[3](h, y_glob) # FiLM y_glob
        h = self.bottleneck[4](h) # Conv2

        for block in self.up_blocks:
            skip_val = skips.pop()
            h = block(h, skip_val, t_emb, y_glob, y_cross)

        out = self.final_conv(h) # (B, D_path, N)
        return out.permute(0, 2, 1) # (B, N, D_path)