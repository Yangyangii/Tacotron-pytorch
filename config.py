
class ConfigArgs:
    data_path = '/home/yangyangii/ssd/data/LJSpeech-1.1'
    mel_dir, mag_dir = 'mels', 'mags'
    meta = 'metadata.csv'
    meta_train = 'meta-train.csv'
    meta_eval = 'meta-eval.csv'
    testset = 'test_sents.txt'
    logdir = 'logs'
    sampledir = 'samples'
    prepro = True
    mem_mode= True
    log_mode = True
    save_term = 1000
    n_workers = 8
    n_gpu = 2
    global_step = 0

    sr = 22050 # sampling rate
    preemph = 0.97 # pre-emphasize
    n_fft = 2048
    n_mags = n_fft//2 + 1
    n_mels = 80
    frame_shift = 0.0125
    frame_length = 0.05
    hop_length = int(sr*frame_shift)
    win_length = int(sr*frame_length)
    gl_iter = 50 # Griffin-Lim iteration
    max_db = 100
    ref_db = 20
    power = 1.2
    r = 5  # reduction factor.
    g = 0.2

    batch_size = 32
    test_batch = 32 # for test
    max_step = 400000
    lr = 0.001
    lr_decay_step = 50000 # actually not decayed per this step
    Ce = 256  # for text embedding 
    Cx = 128 # for context encoding
    Ca = 256 # attention dimension
    drop_rate = 0.05

    max_Tx = 188
    max_Ty = 250

    vocab = u'''PE !',-.?abcdefghijklmnopqrstuvwxyz'''
