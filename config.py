ALL_LABELS = ['NEG', 'POS']

MAX_LENGTH = float('inf')
W2V_DIM = 300
CNN_CONFIG = {
    1: 32,
    2: 64,
    3: 128,
    4: 64,
    5: 32
}
HIDDEN_LAYERS = [128]

DATA = 'data/'
W2V_DATA = DATA + 'w2v/100k_and_train_set_w2v.npz'
VOCAB_DATA = DATA + 'w2v/vocab.txt'
TRAINED_MODELS = DATA + 'trained_models/'
