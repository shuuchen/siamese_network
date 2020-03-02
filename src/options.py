class Config():
    train_file = './config/train_unit_ids.txt'
    val_file = './config/val_unit_ids.txt'
    test_file = './config/test_unit_ids.txt'
    
    data_dir = './data'
    checkpoint_dir = './checkpoint'
    
    network = 'triplet' # 'siamese' or 'triplet'
    
    batch_size = 16
    train_number_epochs = 100