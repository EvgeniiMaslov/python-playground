
class CFG:
    # ------ General ------
    data_path = 'data'
    seed = 1234
    debug = False

    # ------ Data preprocessing ------

    split_ratio = 0.001 if debug else 0.3

    val_size = 0.2
    test_size = 0.1

    # ------ Training ------
    epochs = 1 if debug else 20
    clip = 1