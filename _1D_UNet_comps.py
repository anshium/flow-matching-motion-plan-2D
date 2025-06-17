from imports_self import *

from helper_modules import SinusoidalTimeEmbedding, FiLMLayer, SimpleAttention

# --- 1D U-Net Components ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(min(groups, out_channels) if out_channels >= groups else 1, out_channels)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, y_glob_dim, n_conv=2, self_attn=False, heads=4, dim_head=32):
        super().__init__()
        self.convs = nn.ModuleList([ConvBlock(in_ch if i==0 else out_ch, out_ch) for i in range(n_conv)])
        self.film_t = FiLMLayer(out_ch, t_emb_dim)
        self.film_y = FiLMLayer(out_ch, y_glob_dim)
        self.downsample = nn.Conv1d(out_ch, out_ch, 4, 2, 1)
        self.self_attn = SimpleAttention(out_ch, heads=heads, dim_head=dim_head) if self_attn else None
        if self_attn: self.attn_norm = nn.LayerNorm(out_ch)

    def forward(self, x, t_emb, y_glob):
        for conv in self.convs: x = conv(x)
        x = self.film_t(x, t_emb)
        x = self.film_y(x, y_glob)
        if self.self_attn:
            x_attn = x.permute(0, 2, 1)
            x_attn = self.self_attn(x_attn)
            x = x + x_attn.permute(0, 2, 1)
            x = self.attn_norm(x.permute(0,2,1)).permute(0,2,1)
        skip = x
        return self.downsample(x), skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, y_glob_dim, y_cross_dim=None, n_conv=2, cross_attn=False, heads=4, dim_head=32):
        super().__init__()
        # self.convs = nn.ModuleList([ConvBlock((in_ch + out_ch) if i==0 else out_ch, out_ch) for i in range(n_conv)])
        self.convs = nn.ModuleList([ConvBlock((in_ch // 2 + out_ch) if i==0 else out_ch, out_ch) for i in range(n_conv)])
        self.film_t = FiLMLayer(out_ch, t_emb_dim)
        self.film_y = FiLMLayer(out_ch, y_glob_dim)
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch // 2, 4, 2, 1)
        self.cross_attn = SimpleAttention(out_ch, y_cross_dim, heads, dim_head) if cross_attn and y_cross_dim else None
        if cross_attn and y_cross_dim: self.attn_norm = nn.LayerNorm(out_ch)

    def forward(self, x, skip, t_emb, y_glob, y_cross=None):
        x = self.upsample(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        x = torch.cat([x, skip], dim=1)
        for conv in self.convs: x = conv(x)
        x = self.film_t(x, t_emb)
        x = self.film_y(x, y_glob)
        if self.cross_attn and y_cross is not None:
            x_attn = x.permute(0, 2, 1)
            x_attn = self.cross_attn(x_attn, context=y_cross)
            x = x + x_attn.permute(0, 2, 1)
            x = self.attn_norm(x.permute(0,2,1)).permute(0,2,1)
        return x