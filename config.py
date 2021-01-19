hparams = {
    'batch_size': 50,
    'test_batch_size' : 50,
    'lr': 5e-4,
    'embedding': './word2vec/baomoi.window2.vn.model.bin',
    'model': {
        'dct_size': 'auto',
        'nhead': 10,
        'embed_size': 300,
        # 'self_attn_size': 400,
        'encoder_size': 250,
        'v_size': 200
    },
    'title_len' : 20,
    'his_len' : 30,
    'train': {
        'npratio' : 4,
    },
    'val' : {
        'pos' : 2,
        'neg' : 8,
        'total' : 0
    },
    'test': {
        'pos': 5,
        'neg' : 20,
        'total' : 400
    }
}
