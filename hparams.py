### Audio and feature hparams

#sample_rate=16000
#n_fft = 1024
#hop_length = 512
num_of_frame = 200 # ~30 frames per second.

### Experiment parameters:
kfold_n_splits = 8
random_seeds = [1234, 719, 1011, 129, 1205, 824, 5278, 812, 1231, 9487]
positive_patient_num = 200
negative_patient_num = 200

### Optimization hparams:
positive_negative_loss_ratio = 1
num_workers = 4
num_epochs = 40
batch_size = 64
learning_rate = 0.001
learning_rate_min = 1e-4
weight_decay = 0.001
grad_clip_thresh = 1.0
prob_threshold = 0.5


### Model parameters
model_type = "AttRNN"
complexity = "simple"
# CNN 
if model_type == "AttRNN":
    if complexity == "simple":
        num_convlayers = 2
        in_channels = [1, 64]
        out_channels = [64, 64]
        kernel_size = [(5, 5), (5, 5)]
        padding_size = [(2, 2), (2, 2)]
        pool_size = [(5, 5), (5, 5)]
    elif complexity == "standard":
        num_convlayers = 3
        in_channels = [1, 32, 64]
        out_channels = [32, 64, 128]
        kernel_size = [(3, 5), (3, 5), (3, 5)]
        padding_size = [(1, 2), (1, 2), (1, 2)]
        pool_size = [(2, 2), (5, 2), (5, 5)]
    elif complexity == "large":
        num_convlayers = 4
        in_channels = [1, 64, 64, 128]
        out_channels = [64, 64, 128, 256]
        kernel_size = [(3, 5), (3, 5), (3, 5), (3, 5)]
        padding_size = [(1, 2), (1, 2), (1, 2), (3, 5)]
        pool_size = [(2, 4), (2, 2), (5, 2), (5, 2)]

    # RNN
    num_rnnlayers = 2
    rnn_dropout = 0.25


elif model_type == "AttCNN":
    if complexity == "simple":
        num_convlayers = 2
        in_channels = [1, 32]
        out_channels = [32, 8]
        kernel_size = [(3, 5), (2, 5)]
        pool_size = [(2, 5), (2, 5)]
    elif complexity == "standard":
        num_convlayers = 3
        in_channels = [1, 32, 16]
        out_channels = [32, 16 ,8]
        kernel_size = [(3, 5), (3, 5), (1, 5)]
        pool_size = [(1, 2), (2, 5), (2, 2)]
    elif complexity == "large":
        num_convlayers = 4
        in_channels = [1, 64, 64, 128]
        out_channels = [64, 64, 128, 256]
        kernel_size = [(3, 5), (3, 5), (3, 5), (3, 5)]
        padding_size = [(1, 2), (1, 2), (1, 2), (3, 5)]
        pool_size = [(2, 4), (2, 4), (5, 2), (5, 2)]

cnn_dropout = 0.15

# Attention
att_dropout = 0.0

normalize = "batchnorm"

