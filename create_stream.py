from email.mime import base
from pathlib import Path

import csv
import json
import numpy as np
import os 
import pandas as pd
import random
import sox
import textgrid


def get_all_train_files(target_fpath_list):
    train_files = []
    train_tsv = 'cv-corpus-wav/train.tsv'
    train_df = pd.read_csv(train_tsv, sep='\t').set_index("path", drop = False)

    for fpath in target_fpath_list: 
        fname = Path(fpath).stem + ".wav"
        row_df = train_df.loc[(train_df['path'] == fname)]
        if not row_df.empty:
            train_files.append(fpath)
    
    return train_files


def create_stream(KEYWORD, NUM_OF_CLIPS_PER_CATEGORY, iteration_num, init, use_train_data = True):
    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    clips_dir = Path(base_dir + "cv-corpus-wav/clips/")

    # test_tsv = 'cv-corpus-wav/test.tsv'
    # test_df = pd.read_csv(test_tsv, sep='\t').set_index("path", drop = False)

    train_tsv = 'cv-corpus-wav/train.tsv'
    train_df = pd.read_csv(train_tsv, sep='\t').set_index("path", drop = False)

    # #remove wavs that have been seen to avoid repeats
    # testing_file_name = KEYWORD + "/testing_files.txt"
    # curr_testing_list = []
    # if not init: #not initial stream. remove already seen files.
    #     curr_testing_list = eval(open(testing_file_name, "r").read())
    #     for fpath in curr_testing_list:
    #         fname = os.path.basename(fpath)
    #         test_df = test_df.drop(index=fname)#remove wavs that have been seen to avoid repeats

    #remove wavs that have been seen to avoid repeats
    training_file_name = KEYWORD + "/training_files.txt"
    curr_training_list = []
    if not init: #not initial stream. remove already seen files.
        curr_training_list = eval(open(training_file_name, "r").read())
        for fpath in curr_training_list:
            fname = os.path.basename(fpath)
            train_df = train_df.drop(index=fname)
        # print("len before stream created: ", len(curr_testing_list))

    #extract target and non-target
    # files_w_key = test_df[ test_df['sentence'].str.contains(KEYWORD).fillna(False) ]
    # files_wout_key = test_df[ ~test_df['sentence'].str.contains(KEYWORD).fillna(False) ]

    # # select random wavs for target words
    # targets = list(files_w_key['path'])
    # rand_targets = random.sample(range(0, len(targets)), NUM_OF_CLIPS_PER_CATEGORY)
    # targets = [targets[idx] for idx in rand_targets]
    # target_transcript_list = []
    
    #extract target and non-target
    files_w_key = train_df[ train_df['sentence'].str.contains(KEYWORD).fillna(False) ]
    files_wout_key = train_df[ ~train_df['sentence'].str.contains(KEYWORD).fillna(False) ]

    # select wavs 
    targets = os.listdir(base_dir + KEYWORD + "/target")
    non_targets = os.listdir(base_dir + KEYWORD + "/non_target")

    if use_train_data:
        targets = get_all_train_files(targets)
        non_targets = get_all_train_files(non_targets)

    #select rand targets
    rand_targets = random.sample(range(0, len(targets)), NUM_OF_CLIPS_PER_CATEGORY)
    targets = [targets[idx] for idx in rand_targets]
    target_transcript_list = []

    # select rand non targets
    rand_non_targets = random.sample(range(0, len(non_targets)), NUM_OF_CLIPS_PER_CATEGORY)
    non_targets = [non_targets[idx] for idx in rand_non_targets]
    non_target_transcript_list = []

    # for fname in targets: #populate (target, transcript) list
    #     row = test_df.loc[(test_df['path'] == fname)]
    #     transcript = row['sentence'].values[0].split('\t')[0]
    #     target_transcript_list.append((fname,transcript))
    #     # print("selected targets: ", target_transcript_list)

    
    for fpath in targets: #populate (target, transcript) list
        fname = Path(fpath).stem + ".wav"
        row = train_df.loc[(train_df['path'] == fname)]
        transcript = row['sentence'].values[0].split('\t')[0]
        target_transcript_list.append((fname,transcript))

    # for fname in non_targets: #populate (non-target, transcript) list
    #     row = test_df.loc[(test_df['path'] == fname)]
    #     transcript = row['sentence'].values[0].split('\t')[0]
    #     non_target_transcript_list.append((fname, transcript))
    # print("selected non targets: ", non_target_transcript_list)

    for fpath in non_targets: #populate (non-target, transcript) list
        fname = Path(fpath).stem + ".wav"
        row = train_df.loc[(train_df['path'] == fname)]
        transcript = row['sentence'].values[0].split('\t')[0]
        non_target_transcript_list.append((fname, transcript))

    n_target_wavs = len(target_transcript_list)
    print("num of selected target wavs:", n_target_wavs)
    selected_nontargets = np.random.choice(
        range(len(non_target_transcript_list)), n_target_wavs, replace=False
    )

    workdir = Path(base_dir + KEYWORD + "/v_" + str(iteration_num) + "/streaming")
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    #create the streaming data
    stream_info_file = workdir / "stream_info.pkl"
    assert not os.path.isfile(stream_info_file), "already exists"

    # make streaming wav
    intermediate_wavdir = workdir / "intermediate_wavs"
    if not os.path.exists(intermediate_wavdir):
        os.makedirs(intermediate_wavdir)

    stream_info = []
    stream_wavs = []

    for ix, ((target_wav, target_transcript), nontarget_ix) in enumerate(
        zip(target_transcript_list, selected_nontargets)
    ):
        tw = clips_dir / target_wav
        nw = clips_dir / non_target_transcript_list[nontarget_ix][0] 
        # print ("non target ix: ", nontarget_ix)

        durations_s = []
        # convert all to same samplerate
        for w in [tw, nw]:

            dest = str(intermediate_wavdir / w.name)
            transformer = sox.Transformer()
            transformer.convert(samplerate=16000)  # from 48K mp3s
            transformer.build(str(w), dest)
            stream_wavs.append(dest)
            durations_s.append(sox.file_info.duration(dest))

        # record transcript info
        tw_info = dict(
            ix=2 * ix,
            wav=target_wav,
            transcript=target_transcript,
            duration_s=durations_s[0],
        )
        nw_info = dict(
            ix=2 * ix + 1,
            wav=non_target_transcript_list[nontarget_ix][0],
            transcript=non_target_transcript_list[nontarget_ix][1],
            duration_s=durations_s[1],
        )
        stream_info.extend([tw_info, nw_info])

    assert len(stream_wavs) == n_target_wavs * 2, "not enough stream data"
    stream_wavfile = str(workdir / (KEYWORD + "_stream.wav") )

    #write stream info json
    with open(str(workdir) + "/" + KEYWORD + "_stream_info", "w") as fh:
        json.dump(stream_info, fh)

    combiner = sox.Combiner()
    combiner.convert(samplerate=16000, n_channels=1)
    # https://github.com/rabitt/pysox/blob/master/sox/combine.py#L46
    combiner.build(stream_wavs, stream_wavfile, "concatenate")

    dur_info = sum([d["duration_s"] for d in stream_info])
    print(sox.file_info.duration(stream_wavfile), "seconds in length", dur_info)
        
    # #update testing.txt with new batch of streaming wavs
    # updated_testing_list = curr_testing_list + stream_wavs
    # # print("len after stream created: ", len(updated_testing_list))
    # with open(testing_file_name, "w") as testing_f:
    #     testing_f.write(str(updated_testing_list))

    #update testing.txt with new batch of streaming wavs
    if not init:
        updated_training_list = curr_training_list + stream_wavs
        # print("len after stream created: ", len(updated_testing_list))
        with open(training_file_name, "w") as training_f:
            training_f.write(str(updated_training_list))

    return

#used by train model sweep run
def generate_ground_truth_timings(KEYWORD, iteration_num):
    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    tg_dir = base_dir + KEYWORD + "/data_and_alignments/alignments/"

    gt_target_times_ms = []
    curr_time_s = 0.0

    stream_info_file = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/streaming/" + KEYWORD + "_stream_info"
    stream_info = eval(open(stream_info_file, "r").read())
    
    for segment in stream_info:
        transcript = segment["transcript"]
        contains_keyword = any([w == KEYWORD for w in transcript.split()])
        
        if contains_keyword:
            wav = segment["wav"]

            #find interval where the keyword exists
            tgfile = tg_dir + Path(wav).stem + ".TextGrid"
            tg = textgrid.TextGrid.fromFile(tgfile)

            for interval in tg[0]:
                if interval.mark != KEYWORD:
                    continue
            
                start_s_in_sentence = interval.minTime
                start_s_in_stream = curr_time_s + start_s_in_sentence
                gt_target_times_ms.append(start_s_in_stream * 1000)
        

        #increase curr time before moving on to next segment
        curr_time_s += segment["duration_s"]

    return gt_target_times_ms

#used by batch streaming accuracy
def generate_ground_label_and_times(KEYWORD, iteration_num, stream_info_file):
    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    tg_dir = base_dir + KEYWORD + "/data_and_alignments/alignments/"

    gt_label_and_targets = []
    curr_time_s = 0.0

    stream_info = eval(open(stream_info_file, "r").read())
    
    for segment in stream_info:
        transcript = segment["transcript"]
        contains_keyword = any([w == KEYWORD for w in transcript.split()])
        
        if contains_keyword:
            wav = segment["wav"]

            #find interval where the keyword exists
            tgfile = tg_dir + Path(wav).stem + ".TextGrid"
            tg = textgrid.TextGrid.fromFile(tgfile)

            for interval in tg[0]:
                if interval.mark != KEYWORD:
                    continue
            
                #keyword found
                start_s_in_sentence = interval.minTime
                start_s_in_stream = curr_time_s + start_s_in_sentence
                gt_label_and_targets.append((KEYWORD,start_s_in_stream * 1000))
        

        #increase curr time before moving on to next segment
        curr_time_s += segment["duration_s"]

    #write each keyword to a newline
    gt_fname = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/streaming/" + "ground_truth.txt"
    gt_target_with_newlines = "\n".join(str(label_time_pair[0] + "," + str(label_time_pair[1])) for label_time_pair in gt_label_and_targets)
    with open(gt_fname, "w") as gt_f:
        gt_f.write(gt_target_with_newlines)
    gt_f.close()

    return gt_fname