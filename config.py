from imports_self import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_WAYPOINTS = 20
PATH_DIM = 2

NUM_TRAINING_SAMPLES = 1024 * 10

BATCH_SIZE = 64

OBSTACLE_ENCODER_EMBED_DIM = 128 # Output of global obstacle CNN
OBSTACLE_CONTEXT_TOKENS = 4      # Num tokens for (optional) cross-attention context
OBSTACLE_CONTEXT_DIM = 64        # Dim per token for (optional) cross-attention context
START_GOAL_DIM = PATH_DIM        # Using raw coordinates

UNET_INIT_CONV_CHANNELS = 32
UNET_CHANNEL_MULTS = (1, 2, 4)
UNET_TIME_EMB_DIM = 128

# Global condition dim = start_dim + goal_dim + global_obs_embed_dim
UNET_GLOBAL_COND_DIM = START_GOAL_DIM + START_GOAL_DIM + OBSTACLE_ENCODER_EMBED_DIM
UNET_NUM_CONV_LAYERS = 2
UNET_ATTN_HEADS = 4
UNET_ATTN_DIM_HEAD = 32

# Training params
BATCH_SIZE = 32
N_EPOCHS = 50 * 10 ########### ---> increase for real training
LEARNING_RATE = 1e-4
CFM_SIGMA = 0.01 # Noise level for CFM

# Flags for attention types
USE_SELF_ATTN_IN_UNET = True
USE_CROSS_ATTN_IN_UNET = True # If True, needs y_obstacle_context