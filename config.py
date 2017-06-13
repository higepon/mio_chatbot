# Parameters for Data processing
DATA_PATH = 'data/cornell movie-dialogs corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'

CPT_PATH = 'checkpoints'
PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

THRESHOLD = 2
# Real config parameters to care

BUCKETS = [(8, 10), (12, 14), (16, 19)]
TESTSET_SIZE = 25000
HIDDEN_SIZE = 256
NUM_LAYERS = 3


ENC_VOCAB = 24417
DEC_VOCAB = 24656
