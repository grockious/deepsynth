### The Config ###

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'MontezumaRevenge-v4'

# Saving Learning Progress
# If SAVE_PATH is None, it will not save the agent
SAVE_PATH = 'checkpoint_saves'
SAVE_FRAMES = False

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = 'tensorboard/'

# If True, use the prioritized experience replay algorithm, instead of regular experience replay
USE_PER = True

PRIORITY_SCALE = 0.7               # How much the replay buffer should sample based on priorities. 0 = complete random samples, 1 = completely aligned with priorities
CLIP_REWARD = False                # Any positive reward is +1, and negative reward is -1, 0 is unchanged


TOTAL_FRAMES = int(1e7)           # Total number of frames to train for
FRAMES_BETWEEN_EVAL = 30000       # Number of frames between evaluations
EVAL_LENGTH = 10000               # Number of frames to evaluate for

DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
MIN_REPLAY_BUFFER_SIZE = 50000    # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = 150000                 # The maximum size of the replay buffer
MIN_DFA_FRAMES = 50000            # The minimum number of exploration frames to generate an intial DFA

MAX_NOOP_STEPS = 30               # Randomly perform this number of actions before every evaluation to give it an element of randomness
UPDATE_FREQ = 4                   # Number of actions between gradient descent steps
TARGET_UPDATE_FREQ = 1000         # Number of actions between when the target network is updated
DFA_UPDATE_FREQ = 1000

INPUT_SHAPE = (84, 84)            # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 32                   # Number of samples the agent learns from at once
LEARNING_RATE = 2.5e-4
MU = 5
