
class CFG:
    # ------ General ------
    data_path = 'data'
    seed = 1234
    debug = True

    # ------ Data preprocessing ------
    use_chunks = True
    chunk_size = 10 ** 6

    split_ratio = 0.005 if debug else 0.3

    

    val_size = 0.2
    test_size = 0.1