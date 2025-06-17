from imports_self import *

# --- Helper Modules (Time Embedding, FiLM, Attention) ---
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0,1))
        return embeddings

class FiLMLayer(nn.Module):
    def __init__(self, features_dim, cond_dim):
        super().__init__()
        self.projection = nn.Linear(cond_dim, features_dim * 2)

    def forward(self, features, cond_embedding):
        gamma_beta = self.projection(cond_embedding)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        if features.ndim == 3 and features.shape[1] == gamma.shape[-1]: # (B, C, N)
            gamma = gamma.unsqueeze(-1); beta = beta.unsqueeze(-1)
        elif features.ndim == 3 and features.shape[2] == gamma.shape[-1]: # (B, N, C)
            gamma = gamma.unsqueeze(1); beta = beta.unsqueeze(1)
        elif features.ndim != 2 :
            raise ValueError(f"FiLM features dim {features.shape} not compatible")
        return gamma * features + beta

class SimpleAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x_query, context=None, mask=None):
        b_q, n_q, _ = x_query.shape
        context = context if context is not None else x_query
        b_c, n_c, _ = context.shape

        q = self.to_q(x_query).view(b_q, n_q, self.heads, -1).transpose(1, 2)
        k = self.to_k(context).view(b_c, n_c, self.heads, -1).transpose(1, 2)
        v = self.to_v(context).view(b_c, n_c, self.heads, -1).transpose(1, 2)

        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None: sim = sim.masked_fill(~mask.view(1,1,1,n_c), -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2).reshape(b_q, n_q, -1)
        return self.to_out(out)