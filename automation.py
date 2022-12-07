import create_one_s as cos
import create_stream as cs
import forced_alignment as fa
import train_new_model as tnm
import inference as inf
import os
import post_processing as pp
from pathlib import Path


import glob
import shutil
from collections import Counter
import csv
import pickle
import datetime
from pathlib import Path
import pprint

import pprint
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import sox
import pydub
from pydub.playback import play 

from multilingual_kws.embedding import word_extraction, transfer_learning
from multilingual_kws.embedding import batch_streaming_analysis as sa
from multilingual_kws.embedding.tpr_fpr import tpr_fpr
import textgrid
# from multilingual_kws.luganda.luganda_train import SweepData
# from multilingual_kws.luganda.luganda_info import WavTranscript

KEYWORD = 'abantu'
# KEYWORD = 'gavumenti'
# KEYWORD = 'akawuka'
# KEYWORD = 'wooteri'
# KEYWORD = 'duuka'
# NUM_OF_ITERATIONS = 5
iteration_num = 0
NUM_OF_CLIPS_PER_CATEGORY = 17
# fa.generate_mfas(KEYWORD, NUM_OF_CLIPS_PER_CATEGORY) #-> create text grids for keyword of interest
# cos.create_one_s(KEYWORD)


# setup base model with initial iteration
# tnm.train_model(KEYWORD, iteration_num, [], [], [], init=True)
# iteration_num += 1
# cs.create_stream(KEYWORD, NUM_OF_CLIPS_PER_CATEGORY, iteration_num, init = True)
# inf.inference(keywords=KEYWORD, iteration_num=iteration_num)
# pp.run_post_process(KEYWORD, iteration_num)
##AFTER HUMAN IN THE LOOP
# pp.update_training_txt(KEYWORD, iteration_num)


#generating ROCs
tnm.execute_sweep_run(KEYWORD, iteration_num, [], [], [], init=True)


# abs_path = os.path.abspath(__file__)
# base_dir = os.path.dirname(abs_path) + "/"
# workdir = base_dir + KEYWORD + "/v_" + str(iteration_num)
# exp_dir = workdir/ Path("export")
# mdd = exp_dir / Path("fold_00")

# num_nt = NUM_OF_CLIPS_PER_CATEGORY

# #extract stream duration
# stream_info_file = base_dir + KEYWORD + "/v_" + str(iteration_num) + "/streaming/" + KEYWORD + "_stream_info"
# stream_info = eval(open(stream_info_file, "r").read())
# stream_total_duration = sum([segment['duration_s'] for segment in stream_info])

# ground_truth_timings_ms = cs.generate_ground_truth_timings(KEYWORD, iteration_num)
# eval_data = dict(times=ground_truth_timings_ms, num_nt=num_nt,duration_s=stream_total_duration)

# for exp in os.listdir(exp_dir):
#     rp = exp_dir / mdd / "result.pkl"
#     if not os.path.isfile(rp):
#             continue  # still calculating
#     with open(rp, "rb") as fh:
#             result = pickle.load(fh)
#     with open(mdd / "sweep_data.pkl", "rb") as fh:
#             sweep_data = pickle.load(fh)
#             sweep_info = sweep_data["sweep_datas"]


#     all_tprs = []
#     all_fprs = []
#     all_threshs = []

#     fig, ax = plt.subplots()
#     for post_processing_settings, results_per_thresh in result:
#         for thresh, found_words in results_per_thresh.items():
#             if thresh < 0.3:
#                 continue
#             analysis = tpr_fpr(
#                 KEYWORD,
#                 thresh,
#                 found_words,
#                 eval_data["times"],
#                 duration_s=eval_data["duration_s"],
#                 time_tolerance=post_processing_settings.time_tolerance_ms,
#                 num_nontarget_words=eval_data["num_nt"]
#                 )
#             tpr = analysis["tpr"]
#             fpr = analysis["false_accepts_per_hour"]
#             all_tprs.append(tpr)
#             all_fprs.append(fpr)
#             all_threshs.append(thresh)
#             if np.isclose(thresh, 0.90):
#                         pprint.pprint(analysis)

#         sd = sweep_info[0]
#         if sd.backprop_into_embedding:
#             lrinfo = f"lr1: {sd.primary_lr} lr2: {sd.embedding_lr}"
#         else:
#             lrinfo = f"lr1: {sd.primary_lr}"

#         if sd.with_context:
#             wc = "t"
#         else:
#             wc = "f"

#         num_train = len(sd.train_files)

#         label = f"{KEYWORD} ({iteration_num}-{num_train})"
#         ax.plot(all_fprs, all_tprs, label=label, linewidth=3)

# # ax.axvline(
# #     x=50, label=f"nominal cutoff for false accepts", linestyle="--", color="black",
# # )

# # ax.set_xlim(0, 200)
# # ax.set_ylim(0, 1)
# # ax.legend(loc="lower right")
# # ax.set_xlabel("False Accepts per Hour")
# # ax.set_ylabel("True Positive Rate")
# # for item in (
# #     [ax.title, ax.xaxis.label, ax.yaxis.label]
# #     + ax.get_legend().get_texts()
# #     + ax.get_xticklabels()
# #     + ax.get_yticklabels()
# # ):
# #     item.set_fontsize(25)
# # fig.set_size_inches(14, 14)
# # fig.tight_layout()
# # figdest = "base_dir/luganda_" + KEYWORD + ".png"
# # fig.savefig(figdest)
# # print(figdest)



