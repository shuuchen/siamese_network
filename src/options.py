class Config():
    train_file = '/home/ec2-user/github/siamese_network/config/train_unit_ids.txt'
    val_file = '/home/ec2-user/github/siamese_network/config/val_unit_ids.txt'
    test_file = '/home/ec2-user/github/siamese_network/config/test_unit_ids.txt'
    
    data_dir = '/home/ec2-user/github/siamese_network/data'
    checkpoint_dir = '/home/ec2-user/github/siamese_network/checkpoint'
    
    network = 'triplet' # 'siamese' or 'triplet'
    
    batch_size = 16
    train_number_epochs = 100