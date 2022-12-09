import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("/media/cbanbury/T7/jquaye/multilingual_kws/")

from pathlib import Path
import glob
import shutil
from collections import Counter
import create_stream as cs
import csv
import datetime
import _json
import pickle
import pprint


from multilingual_kws.embedding import batch_streaming_analysis as sa
from multilingual_kws.embedding.batch_streaming_analysis import StreamTarget, StreamFlags
from multilingual_kws.embedding.tpr_fpr import tpr_fpr
import textgrid
from train_new_model import SweepData

def generate_stats(KEYWORD, iteration_num, NUM_OF_CLIPS_PER_CATEGORY):
    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    workdir = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/"
    exp_dir = workdir / Path("export") / Path("exp_01")
    mdd = exp_dir / f"fold_00"

    num_nt = NUM_OF_CLIPS_PER_CATEGORY

    #extract stream duration
    stream_info_file = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/streaming/" + KEYWORD + "_stream_info"
    stream_info = eval(open(stream_info_file, "r").read())
    stream_total_duration = sum([segment['duration_s'] for segment in stream_info])

    ground_truth_timings_ms = cs.generate_ground_truth_timings(KEYWORD, iteration_num)
    eval_data = dict(times=ground_truth_timings_ms, num_nt=num_nt,duration_s=stream_total_duration)

    # fig, ax = plt.subplots()

    for exp in os.listdir(exp_dir):
        rp = mdd / "result.pkl"
        if not os.path.isfile(rp):
            print (rp)
            print("given result does not exist")
            continue  # still calculating
        with open(rp, "rb") as fh:
                result = pickle.load(fh)

        sweep_data = eval(open(mdd / "sweep_data.txt", "r").read())
        sweep_info = sweep_data["sweep_datas"]

        all_tprs = []
        all_fprs = []
        all_frhs = []
        all_fahs = []
        all_tahs = []
        all_threshs = []

        # fig, ax = plt.subplots()
        for post_processing_settings, results_per_thresh in result[KEYWORD]:
            for thresh, (found_words,_) in results_per_thresh.items():
                if thresh < 0.3:
                    continue
                analysis = tpr_fpr(
                    KEYWORD,
                    thresh,
                    found_words,
                    gt_target_times_ms=eval_data["times"],
                    duration_s=eval_data["duration_s"],
                    time_tolerance_ms=post_processing_settings.time_tolerance_ms,
                    num_nontarget_words=eval_data["num_nt"]
                    )
                
                stream_len = eval_data["duration_s"]
                gt_target_times_ms = eval_data["times"]

                true_positives = analysis["true_positives"]
                true_negatives = 2 * len(gt_target_times_ms)  #consider true negative examples to be twice as long as false positives
                false_positives = analysis["false_positives"]
                false_negatives = analysis["false_negatives"]

                tpr = true_positives / len(gt_target_times_ms)
                fpr = false_positives / (false_positives + true_negatives )

                fah = false_positives / (stream_len * 3600) #converting to false accepts (positives) per hour
                frh = false_negatives / (stream_len * 3600) #converting to false rejects (positives) per hour
                tah = true_positives / (stream_len * 3600) #converting to true positives per hour

                all_tprs.append(tpr)
                all_fprs.append(fpr)
                all_fahs.append(fah)
                all_frhs.append(frh)
                all_tahs.append(tah)

                all_threshs.append(thresh)
                if np.isclose(thresh, 0.90):
                    pprint.pprint(analysis)

                

            sd = sweep_info[0]
            if sd.backprop_into_embedding:
                lrinfo = f"lr1: {sd.primary_lr} lr2: {sd.embedding_lr}"
            else:
                lrinfo = f"lr1: {sd.primary_lr}"

            if sd.with_context:
                wc = "t"
            else:
                wc = "f"

            num_train = len(sd.train_files)

            sample_dur = eval_data["duration_s"]

            post_proc_data = {}
            post_proc_data["all_tprs"] = all_tprs
            post_proc_data["all_fprs"] = all_fprs
            post_proc_data["all_fahs"] = all_fahs
            post_proc_data["all_frhs"] = all_frhs
            post_proc_data["all_tahs"] = all_tahs 
            post_proc_data["all_threshs"] = all_threshs
            post_proc_data["sample_dur"] = sample_dur 

            #write to a file for this version
            stats_file = workdir + "stats_file.txt"
            with open(stats_file, "w") as stats_f:
                stats_f.write(str(post_proc_data))
            stats_f.close()

            label = f"{KEYWORD} (iteration-{iteration_num})"
            print("all_fprs: ", all_fprs)
            print("all tprs: ", all_tprs)
            print("all false accepts per hour: ", all_fahs)
            print("all false rejects per hour: ", all_frhs)
            print("all true positives per hour: ", all_tahs)
            
            plt.rcParams.update({'font.size':20})
            plt.figure(figsize=(14,14))
            plt.ylim(0,1)
            plt.xlim(0,1)
            plt.plot(all_fprs, all_tprs, label=label, linewidth=3)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            roc_figdest = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/ROC.png"
            plt.savefig(roc_figdest)
            # plt.show()

            plt.figure(figsize=(14,14))
            plt.plot(all_threshs, all_fahs, label=label, linewidth=3)
            plt.ylabel("False Accepts Per Hour")
            plt.xlabel("Threshold")
            plt.legend()
            fahs_figdest = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/fahs.png"
            plt.savefig(fahs_figdest)
            # plt.show()

            plt.figure(figsize=(14,14))
            plt.plot(all_threshs, all_frhs, label=label, linewidth=3)
            plt.ylabel("False Rejects Per Hour")
            plt.xlabel("Threshold")
            plt.legend()
            frhs_figdest = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/frhs.png"
            plt.savefig(frhs_figdest)
            # plt.show()

            plt.figure(figsize=(14,14))
            plt.plot(all_threshs, all_tahs, label=label, linewidth=3)
            plt.ylabel("True Accepts Per Hour")
            plt.xlabel("Threshold")
            plt.legend()
            tahs_figdest = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/tahs.png"
            plt.savefig(tahs_figdest)
            plt.show()