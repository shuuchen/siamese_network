class Config():

    labelling = True

    if labelling:
        train_file = '/home/ec2-user/github/siamese_network/config/labelling/train_unit_ids.txt'
        val_file = '/home/ec2-user/github/siamese_network/config/labelling/val_unit_ids.txt'
        test_file = '/home/ec2-user/github/siamese_network/config/labelling/test_unit_ids.txt'
    else:
        train_file = '/home/ec2-user/github/siamese_network/config/non-labelling/train_unit_ids.txt'
        val_file = '/home/ec2-user/github/siamese_network/config/non-labelling/val_unit_ids.txt'
        test_file = '/home/ec2-user/github/siamese_network/config/non-labelling/test_unit_ids.txt'
   
    data_dir = '/home/ec2-user/github/siamese_network/data'
    checkpoint_dir = '/home/ec2-user/github/siamese_network/checkpoint'
    
    network = 'triplet' # 'siamese' or 'triplet'
    
    batch_size = 16
    train_number_epochs = 100
