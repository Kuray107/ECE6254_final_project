### Audio and feature hparams
sample_rate=16000
n_fft = 1024
hop_length = 512
num_of_frame = 150 # ~30 frames per second.
positive_patient_num = 40
negative_patient_num = 40

### Optimization hparams:
num_workers = 1
num_epochs = 30
batch_size = 32
learning_rate = 0.001
weight_decay = 0.001
grad_clip_thresh = 1.0




### Model parameters
normalize = None
input_feature = "all" # Use both mel and mfcc

