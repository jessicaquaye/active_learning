import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #suppress TF warnings
import sys
sys.path.append("/media/cbanbury/T7/jquaye/multilingual_kws/")
from multilingual_kws.embedding import transfer_learning, input_data

def train_model(KEYWORD, iteration_num, train_samples, dev_samples, non_target_samples, init):

    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"

    #set up background
    resources_dir = base_dir + "resources/"
    background_noise = resources_dir + "speech_commands/_background_noise_/"

    if init: #initial model. generate samples and load old input
        NUM_TRAIN_SAMPLES = 5
        one_s_path = base_dir + KEYWORD + "/one_s/"
        target_fnames = os.listdir(one_s_path)
        rng = np.random.RandomState(0) #sort FS listings to aid reproducibility
        rand_train_samples = rng.choice(target_fnames, NUM_TRAIN_SAMPLES, replace=False)
        train_paths = [ one_s_path + fname for fname in rand_train_samples]
        rand_dev_samples = rng.choice(target_fnames, NUM_TRAIN_SAMPLES, replace=False)
        dev_paths = [ one_s_path + fname for fname in rand_dev_samples]

        #write train paths to file
        with open(KEYWORD + "/training_files.txt", "w") as train_f:
            train_f.write(str(train_paths))

        #write dev paths to file
        with open(KEYWORD + "/dev_files.txt", "w") as dev_f:
            dev_f.write(str(dev_paths))

        #read unknown files from resource and write to .txt file for this iteration
        unknown_files=[]
        unknown_files_txt = resources_dir + "unknown_files/unknown_files.txt"
        with open(unknown_files_txt, "r") as fh:
            for w in fh.read().splitlines():
                unknown_files.append(resources_dir + "unknown_files/" + w)
        print("Number of unknown files", len(unknown_files))

        with open(KEYWORD + "/non_target_files.txt", "w") as non_target_f:
            non_target_f.write(str(unknown_files))

        #define base_model_path
        base_model_path = resources_dir + "embedding_model/multilingual_context_73_0.8011"

    else: #iteration building upon previous model

        #define new base model path 
        base_model_path = 0 #previous base model path

        #read previous train paths and update with curr train paths
        prev_train_content = open(KEYWORD + "/training_files.txt", "r").read()
        prev_train_paths = eval(prev_train_content)
        train_paths = prev_train_paths + train_samples

        with open(KEYWORD + "/training_files.txt", "w") as train_f:
            train_f.write(str(train_paths))
        
        #read previous dev paths and update with curr dev paths
        prev_dev_content = open(KEYWORD + "/dev_files.txt", "r").read()
        prev_dev_paths = eval(prev_dev_content)
        dev_paths = prev_dev_paths + dev_samples

        with open(KEYWORD + "/dev_files.txt", "w") as dev_f:
            dev_f.write(str(dev_paths))

        #read prev unknown paths and update with curr non_target paths
        prev_non_target_content = open(KEYWORD + "/non_target_files.txt", "r").read()
        prev_non_target_paths = eval(prev_non_target_content)
        non_target_paths = prev_non_target_paths + non_target_samples

        with open(KEYWORD + "/non_target_files.txt", "w") as unknown_f:
            unknown_f.write(str(non_target_paths))

        #select previous model as base for training with a new data input
        base_model_path = base_dir + KEYWORD + "/v_" + str(iteration_num - 1) + "/" + KEYWORD + "_5shot"

    print("---Training model---")
    model_settings = input_data.standard_microspeech_model_settings(3)
    _, model, _ = transfer_learning.transfer_learn(
        target=KEYWORD,
        train_files=train_paths,
        val_files=dev_paths,
        unknown_files=unknown_files,
        num_epochs=4,
        num_batches=1,
        batch_size=64,
        primary_lr=0.001,
        backprop_into_embedding=False,
        embedding_lr=0,
        model_settings=model_settings,
        base_model_path=base_model_path,
        base_model_output="dense_2",
        UNKNOWN_PERCENTAGE=50.0,
        bg_datadir=background_noise,
        csvlog_dest=None,
    )

    # os.makedirs(base_dir + KEYWORD + "/v_" + str(iteration_num))
    model.save(base_dir + KEYWORD + "/v_" + str(iteration_num) + "/" + KEYWORD + "_5shot")

    return




