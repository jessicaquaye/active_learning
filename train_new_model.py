import create_stream as cs
import datetime
import json
import multiprocessing
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #suppress TF warnings
import sys
sys.path.append("/media/cbanbury/T7/jquaye/multilingual_kws/")

from dataclasses import dataclass, asdict
from pathlib import Path
from multilingual_kws.embedding import transfer_learning, input_data
from multilingual_kws.embedding import batch_streaming_analysis as sa
from typing import List


def train_model(KEYWORD, iteration_num, train_samples, dev_samples, non_target_samples, init):

    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"

    #set up background
    resources_dir = base_dir + "resources/"
    background_noise = resources_dir + "speech_commands/_background_noise_/"

    if init: #initial model. generate samples and load old input
        # NUM_TRAIN_SAMPLES = 540
        # NUM_DEV_SAMPLES = 48
        # one_s_path = base_dir + KEYWORD + "/one_s/"
        # target_fnames = os.listdir(one_s_path)
        # rng = np.random.RandomState(0) #sort FS listings to aid reproducibility
        # rand_train_samples = rng.choice(target_fnames, NUM_TRAIN_SAMPLES, replace=False)
        # train_paths = [ one_s_path + fname for fname in rand_train_samples]
        # rand_dev_samples = rng.choice(target_fnames, NUM_DEV_SAMPLES, replace=False)
        # dev_paths = [ one_s_path + fname for fname in rand_dev_samples]

        NUM_TRAIN_SAMPLES = 10
        NUM_DEV_SAMPLES = 10
        one_s_path = base_dir + KEYWORD + "/one_s/"
        target_fnames = os.listdir(one_s_path)
        train_paths = [one_s_path + fname for fname in target_fnames[:NUM_TRAIN_SAMPLES]]
        dev_paths = [one_s_path + fname for fname in target_fnames[NUM_TRAIN_SAMPLES:NUM_DEV_SAMPLES+NUM_TRAIN_SAMPLES]]

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
        base_model_output = "dense_2"

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
        unknown_files = prev_non_target_paths + non_target_samples

        with open(KEYWORD + "/non_target_files.txt", "w") as unknown_f:
            unknown_f.write(str(unknown_files))

        #select previous model as base for training with a new data input
        # base_model_path = base_dir + KEYWORD + "/v_" + str(iteration_num - 1) + "/" + KEYWORD + "_5shot"
        # base_model_output = "dense_1"
        base_model_path = resources_dir + "embedding_model/multilingual_context_73_0.8011"
        base_model_output = "dense_2"

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
        base_model_output=base_model_output,
        UNKNOWN_PERCENTAGE=50.0,
        bg_datadir=background_noise,
        csvlog_dest=None,
    )

    # os.makedirs(base_dir + KEYWORD + "/v_" + str(iteration_num))
    model.save(base_dir + KEYWORD + "/v_" + str(iteration_num) + "/" + KEYWORD + "_5shot")

    return


@dataclass(frozen=True)
class SweepData:
    train_files: List[os.PathLike]
    val_files: List[os.PathLike]
    n_batches: int
    n_epochs: int
    model_dest_dir: os.PathLike
    dest_pkl: os.PathLike
    dest_inf: os.PathLike
    primary_lr: float
    backprop_into_embedding: bool
    embedding_lr: float
    with_context: bool
    target: str
    stream_target: sa.StreamTarget
    batch_size: int = 64

    def create_dict(self):
        sd_dict = {}
        sd_dict['train_files'] = [str(file) for file in self.train_files]
        sd_dict['val_files'] = [str(file) for file in self.val_files]
        sd_dict['n_batches'] = self.n_batches
        sd_dict['n_epochs'] = self.n_epochs
        sd_dict['model_dest_dir'] = str(self.model_dest_dir)
        sd_dict['dest_pkl'] = str(self.dest_pkl)
        sd_dict['dest_inf'] = str(self.dest_inf)
        sd_dict['primary_lr'] = self.primary_lr
        sd_dict['backprop_into_embedding'] = self.backprop_into_embedding
        sd_dict['embedding_lr'] = self.embedding_lr
        sd_dict['with_context'] = self.with_context
        sd_dict['target'] = self.target
        sd_dict['stream_target'] = self.stream_target
        sd_dict['batch_size'] = self.batch_size
        return sd_dict


def sweep_run(sd: SweepData, q, iteration_num, init, non_target_samples):
    # load embedding model
    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    resources_dir = base_dir + "resources/"
    background_noise = resources_dir + "speech_commands/_background_noise_/"

    KEYWORD = sd.target

    if init:
        unknown_files=[]
        unknown_files_txt = resources_dir + "unknown_files/unknown_files.txt"
        with open(unknown_files_txt, "r") as fh:
            for w in fh.read().splitlines():
                unknown_files.append(resources_dir + "unknown_files/" + w)

        with open(KEYWORD + "/non_target_files.txt", "w") as non_target_f:
            non_target_f.write(str(unknown_files))

        #define base_model_path
        base_model_path = resources_dir + "embedding_model/multilingual_context_73_0.8011"

    else: #iteration building upon previous model
        #read prev unknown paths and update with curr non_target paths
        prev_non_target_content = open(KEYWORD + "/non_target_files.txt", "r").read()
        prev_non_target_paths = eval(prev_non_target_content)
        # print(type(prev_non_target_paths), prev_non_target_paths)
        # print(type(non_target_samples), non_target_samples)
        unknown_files = prev_non_target_paths + non_target_samples

        with open(KEYWORD + "/non_target_files.txt", "w") as unknown_f:
            unknown_f.write(str(unknown_files))

        #select previous model as base for training with a new data input
        print(sd.model_dest_dir)
        # base_model_path = base_dir + KEYWORD + "/v_" + str(iteration_num - 1) + "/export/exp_01/fold_00/xfer_epochs_4_bs_64_nbs_2_val_acc_0.75_target_" + KEYWORD
        # base_model_path = base_dir + KEYWORD + "/v_" + str(iteration_num - 1) + "/" + KEYWORD + "_5shot"
        base_model_path = resources_dir + "embedding_model/multilingual_context_73_0.8011"


    model_settings = input_data.standard_microspeech_model_settings(3)
    name, model, details = transfer_learning.transfer_learn(
        target=sd.target,
        train_files=sd.train_files,
        val_files=sd.val_files,
        unknown_files=unknown_files,
        num_epochs=sd.n_epochs,
        num_batches=sd.n_batches,
        batch_size=sd.batch_size,
        primary_lr=sd.primary_lr,
        backprop_into_embedding=False,
        embedding_lr=sd.embedding_lr,
        model_settings=model_settings,
        base_model_path=base_model_path,
        base_model_output="dense_2",
        UNKNOWN_PERCENTAGE=50.0,
        bg_datadir=background_noise,
        csvlog_dest=sd.model_dest_dir / "log.csv",
    )
    print("saving", name)
    modelpath = sd.model_dest_dir / name
    model.save(modelpath)

    specs = [input_data.file2spec(model_settings, f) for f in sd.val_files]
    specs = np.expand_dims(specs, -1)
    preds = model.predict(specs)
    amx = np.argmax(preds, axis=1)
    # print(amx)
    val_accuracy = amx[amx == 2].shape[0] / preds.shape[0]
    print("VAL ACCURACY", val_accuracy)

    start = datetime.datetime.now()
    sa.eval_stream_test(sd.stream_target, live_model=model) #needed to put it in a list to serialize
    end = datetime.datetime.now()
    print("time elapsed (for all thresholds)", end - start)

    q.put(val_accuracy)
    return

def execute_sweep_run(KEYWORD, iteration_num, train_samples, dev_samples, non_target_samples, init):

    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    workdir = base_dir + KEYWORD + "/v_" + str(iteration_num)

    #set up background
    resources_dir = base_dir + "resources/"
    background_noise = resources_dir + "speech_commands/_background_noise_/"


    if init: #initial model. generate samples and load old input
        NUM_TRAIN_SAMPLES = 540
        NUM_DEV_SAMPLES = 48
        one_s_path = base_dir + KEYWORD + "/one_s/"
        target_fnames = os.listdir(one_s_path)
        train_paths = [one_s_path + fname for fname in target_fnames[:NUM_TRAIN_SAMPLES]]
        dev_paths = [one_s_path + fname for fname in target_fnames[NUM_TRAIN_SAMPLES:NUM_DEV_SAMPLES+NUM_TRAIN_SAMPLES]]

        #write train paths to file
        with open(KEYWORD + "/training_files.txt", "w") as train_f:
            train_f.write(str(train_paths))

        #write dev paths to file
        with open(KEYWORD + "/dev_files.txt", "w") as dev_f:
            dev_f.write(str(dev_paths))


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

    t = train_paths
    v = dev_paths

    print("---------NUM TRAINING SAMPLES\n", len(train_paths))
    streaming_wav = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/streaming/" + KEYWORD + "_stream.wav" 
    assert os.path.isfile(streaming_wav), "no stream wav"

    q = multiprocessing.Queue()
    val_accuracies = []
    sweep_datas = []

    exp_dir = workdir / Path("export") / Path("exp_01")
    os.makedirs(exp_dir, exist_ok=False)

    for ix in range(1):

        mdd = exp_dir / f"fold_{ix:02d}"
        dp = mdd / "result.pkl"
        di = mdd / "inferences.npy"
        print(mdd)
        os.makedirs(mdd)

        stream_folder = os.path.dirname(streaming_wav) + "/"
        stream_info_file = stream_folder / Path(KEYWORD + "_stream_info")
        print("stream info file: ", stream_info_file)
        gt = cs.generate_ground_label_and_times(KEYWORD, iteration_num, stream_info_file)

        print("ground truth created at:", gt)

        flags = sa.StreamFlags(
            wav=str(streaming_wav),
            ground_truth=str(gt),
            target_keyword=KEYWORD,
            detection_thresholds=np.linspace(
                0.05, 1, 20
            ).tolist(),  # step threshold 0.05
            average_window_duration_ms=100,
            suppression_ms=500,
            time_tolerance_ms=750, #only used when graphing
        )

        streamtarget = sa.StreamTarget(
            target_lang="lu",
            target_word=KEYWORD,
            model_path=None,  # dont save model
            destination_result_pkl=dp,
            destination_result_inferences=di,
            stream_flags=[flags],
        )

        sd = SweepData(
            train_files=t,
            val_files=v,
            n_batches=2,
            batch_size=64,
            n_epochs=4,
            model_dest_dir=mdd,
            dest_pkl=dp,
            dest_inf=di,
            primary_lr=0.001,
            backprop_into_embedding=False,
            embedding_lr=0.00001,
            with_context=True,
            target=KEYWORD,
            stream_target=streamtarget,
        )

        start = datetime.datetime.now()
        p = multiprocessing.Process(target=sweep_run, args=(sd, q, iteration_num, init, non_target_samples))
        p.start()
        p.join()
        end = datetime.datetime.now()
        print("\n\n::::::: experiment run elapsed time", end - start, "\n\n")

        val_acc = q.get()
        val_accuracies.append(val_acc)

        sweep_datas.append(sd)

        # overwrite on every experiment
        # with open(mdd / "sweep_data.pkl", "wb") as fh:
        #     d = dict(sweep_datas=sweep_datas, val_accuracies=val_accuracies)
        #     fh.write(json.dumps(d))

        with open(mdd / "sweep_data.txt", "w") as fh:
            d = dict(sweep_datas=sweep_datas, val_accuracies=val_accuracies)
            fh.write(str(d))
        fh.close()
            
    return