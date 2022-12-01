

def setup_training_files(KEYWORD):
    #set up iterations doc
    iterations_file = Path(KEYWORD + "/num_iterations.txt")
    NUM_OF_ITERATIONS = 5
    with open(iterations_file, 'w') as f:
        f.write("[]")
    f.close()

    # training_file = Path(KEYWORD + "/training_files.txt")
    # with open(training_file, 'w') as train_f:
    #     train_f.write("[]")
    # train_f.close()

    # dev_file = Path(KEYWORD + "/dev_files.txt")
    # with open(dev_file, 'w') as dev_f:
    #     f.write("[]")
    # f.close()

    # testing_file = Path(KEYWORD + "/testing_files.txt")
    # with open(testing_file, 'w') as f:
    #     f.write("[]")
    # f.close()

    #set up unknown files
    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"

    
