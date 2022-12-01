import create_one_s as cos
import create_stream as cs
import forced_alignment as fa
import train_new_model as tnm
import inference as inf
import os
import post_processing as pp
from pathlib import Path

KEYWORD = 'abantu'
NUM_OF_CLIPS_PER_CATEGORY = 10
NUM_OF_ITERATIONS = 5
# fa.generate_mfas(KEYWORD, NUM_OF_CLIPS_PER_CATEGORY)
# cos.create_one_s(KEYWORD)

iteration_num = 0

# #setup base model with initial iteration
# tnm.train_model(KEYWORD, iteration_num, [], [], [], init=True)
# iteration_num += 1

# cs.create_stream(KEYWORD, NUM_OF_CLIPS_PER_CATEGORY, iteration_num, init = True)
# inf.inference(keywords=KEYWORD, iteration_num=iteration_num)
# pp.run_post_process(KEYWORD, iteration_num)

#to update training sample, go into detectons

# train_samples = 
# dev_samples = 
# non_target_samples = 

# for i in range(1, NUM_OF_ITERATIONS):
    # train model
    
    # create stream
    # run inference
