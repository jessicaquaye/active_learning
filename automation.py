# import create_one_s as cos
# import create_stream as cs
# import forced_alignment as fa
# import generate_graphs as gg
# import train_new_model as tnm
# import inference as inf
import os
import matplotlib.pyplot as plt

# import post_processing as pp

# KEYWORD = 'abantu'
# KEYWORD = 'gavumenti'
# KEYWORD = 'akawuka'
# KEYWORD = 'wooteri'
# KEYWORD = 'duuka'
# NUM_OF_ITERATIONS = 5
# iteration_num = 1
# NUM_OF_CLIPS_PER_CATEGORY = 10
# fa.generate_mfas(KEYWORD, NUM_OF_CLIPS_PER_CATEGORY) #-> create text grids for keyword of interest
# cos.create_one_s(KEYWORD)

# cs.create_stream(KEYWORD, NUM_OF_CLIPS_PER_CATEGORY, iteration_num, init = False, use_train_data=False)
# tnm.train_model(KEYWORD, iteration_num, [], [], [], init=True)
# inf.inference(keywords=KEYWORD, iteration_num=iteration_num)

#generating ROCs
# tnm.execute_sweep_run(KEYWORD, iteration_num, [], [], [], init=False)
# gg.generate_stats(KEYWORD, iteration_num, NUM_OF_CLIPS_PER_CATEGORY)

# pp.run_post_process(KEYWORD, iteration_num)
##AFTER HUMAN IN THE LOOP
# pp.update_training_txt(KEYWORD, iteration_num)



#draw final graph
abs_path = os.path.abspath(__file__)
base_dir = os.path.dirname(abs_path) + "/"
KEYWORD = 'abantu'

plt.figure(figsize=(14,14))
plt.rcParams.update({'font.size':20})



# plt.xlabel("Model Confidence")
# plt.ylabel("True Positive Rate")

# for i in [0,8,16,22]:
# # for i in range(23):
#     #get the stats file
#     fpath = base_dir + KEYWORD + "/v_" + str(i) + "/stats_file.txt"
    
#     if os.path.exists(fpath):
#         stat_content = eval(open(fpath, "r").read())
#         all_tprs = stat_content['all_tprs']
#         all_threshs = stat_content['alll_threshs']

#         plt.plot(all_threshs, all_tprs, label="Iteration " + str(i), linewidth=3)
        
#         plt.legend()
# plt.show()
        

# plt.xlabel("Model Confidence")
# plt.ylabel("False Accepts Per Hour")

# # for i in range(23):
# for i in [0, 8, 16, 20 ]:
#     #get the stats file
#     fpath = base_dir + KEYWORD + "/v_" + str(i) + "/stats_file.txt"
    
#     if os.path.exists(fpath):
#         stat_content = eval(open(fpath, "r").read())
#         all_fahs = stat_content['all_fahs']
#         all_threshs = stat_content['alll_threshs']

#         plt.plot(all_threshs, all_fahs, label="Iteration " + str(i), linewidth=3)
        
#         plt.legend()
# plt.show()
        
plt.xlabel("Model Confidence")
plt.ylabel("False Rejects Per Hour")

# for i in range(23):
for i in [0, 8, 16, 20 ]:
    #get the stats file
    fpath = base_dir + KEYWORD + "/v_" + str(i) + "/stats_file.txt"
    
    if os.path.exists(fpath):
        stat_content = eval(open(fpath, "r").read())
        all_frhs = stat_content['all_frhs']
        all_threshs = stat_content['alll_threshs']

        plt.plot(all_threshs, all_frhs, label="Iteration " + str(i), linewidth=3)
        
        plt.legend()
plt.show()
        

































## setup base model with initial iteration
# cs.create_stream(KEYWORD, NUM_OF_CLIPS_PER_CATEGORY, iteration_num, init = True, use_train_data=True)
# tnm.train_model(KEYWORD, iteration_num, [], [], [], init=True)
# inf.inference(keywords=KEYWORD, iteration_num=iteration_num)