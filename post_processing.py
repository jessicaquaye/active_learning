import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #suppress TF warnings
import sox
import sys
sys.path.append("/media/cbanbury/T7/jquaye/multilingual_kws/")

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from multilingual_kws import run
from pydub import AudioSegment
from pydub.playback import play

def trim_audio(input_file, dest_file, start_s, end_s):
    tfm = sox.Transformer()
    tfm.convert(samplerate=16000)
    tfm.trim(start_s, end_s)
    tfm.build(input_file, dest_file)
    return

def extract_keyword_intervals_from_stream(KEYWORD, detections, stream_info_path):
    f = open(stream_info_path, 'r')
    stream_data = json.load(f)
    keyword_intervals = []
    start_s = 0

    for sample in stream_data:
        fname = sample['wav']
        transcript = sample['transcript']
        duration = sample['duration_s']
        end_s = start_s + duration      

        if KEYWORD in transcript:
            keyword_intervals.append( (fname, start_s,end_s) )
        start_s = end_s 
    print (keyword_intervals)
    return keyword_intervals

def classify_detections(stream_wavfile, detections, keyword_intervals, target_dest, unknown_dest):
    stream_len = sox.file_info.duration(stream_wavfile)

    #for each detection, check if it falls within the keyword intervals
    for det in detections:
        start_ms = det['time_ms']
        start_s = start_ms / 1000

        #iterate through intervals to compare
        interval_found = False
        dest_path = unknown_dest + str(start_ms) + ".wav"
        for fname,start,end in keyword_intervals:
            end_s = start_s + 1
            if end_s > stream_len:
                end_s = stream_len

            if start <= start_s <= end: #target word found, add to nc_target
                dest_path = target_dest + Path(fname).stem + "_" + str(start_ms) + ".wav"
                interval_found = True 
                trim_audio(stream_wavfile, dest_path , start_s, end_s)   
                break
          
        if not interval_found:
            trim_audio(stream_wavfile, dest_path , start_s, end_s)
    return

def run_post_process(KEYWORD, iteration_num):

    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    stream_wavfile = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/streaming/" + KEYWORD + "_stream.wav"

    detection_fname = base_dir + KEYWORD + "/v_" + str(iteration_num) + '/detections.json'
    f = open(detection_fname)
    detections = json.load(f)["detections"]
    f.close()

    stream_info_path = base_dir + KEYWORD + '/v_' + str(iteration_num) + '/streaming/' + KEYWORD + '_stream_info'
    keyword_intervals = extract_keyword_intervals_from_stream(KEYWORD, detections, stream_info_path)

    #based on detections and known intervals, classify detections into target and unknowns 
    results_dir = base_dir + KEYWORD +  '/v_' + str(iteration_num) + '/categorized_results/'
    nc_target = results_dir + 'targets/'
    nc_non_target = results_dir + 'non_targets/'
    os.makedirs(nc_target)
    os.makedirs(nc_non_target)
    classify_detections(stream_wavfile, detections, keyword_intervals, nc_target, nc_non_target)

    return

def update_training_txt(KEYWORD, iteration_num):

    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    results_dir = base_dir + KEYWORD +  '/v_' + str(iteration_num) + '/categorized_results/'
    nc_target = results_dir + 'targets/'
    nc_non_target = results_dir + 'non_targets/'

    #update training files with targets
    train_path = base_dir + KEYWORD + "/training_files.txt"
    train_content = eval(open(train_path, "r").read())
    updated_train = train_content + os.listdir(nc_target)

    with open(train_path, "w") as train_f:
        train_f.write(updated_train)

    #update unknown files with non_targets
    non_target_path = base_dir + KEYWORD + "/non_target_files.txt"
    non_target_content = eval(open(non_target_path, "r").read())
    updated_non_target = non_target_content + os.listdir(nc_non_target)

    with open(non_target_path, "w") as non_target_f:
        non_target_f.write(updated_non_target)

    return