import os
import pandas as pd
import shutil
import string
from pathlib import Path

def populate_t_and_nt_audios(df, clips_dir, target_clips, non_target_clips, KEYWORD, NUM_OF_CLIPS_PER_CATEGORY):
    #extract target and non-target
    files_w_key = df[ df['sentence'].str.contains(KEYWORD).fillna(False) ]
    files_wout_key = df[ ~df['sentence'].str.contains(KEYWORD).fillna(False) ]

    #select 50 target and non-target that will be used for test
    target = list(files_w_key['path'])[:NUM_OF_CLIPS_PER_CATEGORY]
    non_target = list(files_wout_key['path'])[:NUM_OF_CLIPS_PER_CATEGORY]

    # print("target clips size: ", len(target))
    # print("non target clips size: ", len(non_target))

    #copy audio clips into respective folders
    print("iterating through target files")
    for fname in target:

        shutil.copy2(clips_dir / fname, target_clips / fname)
    print("iterating through non_target files")
    for fname in non_target:
        shutil.copy2(clips_dir / fname, non_target_clips / fname)
    return [(target_clips, target), (non_target_clips, non_target)]

def create_word_labs(df, KEYWORD, data_and_aligns, clips_of_interest_list, clips_dir):
    written = 0

    all_words_in_src = set()

    for fname in clips_of_interest_list:
        row = df.loc[(df['path'] == fname)]
        sentence = row['sentence'].values[0]
        transcript = sentence.split('\t')[0] #some of the sentences contained '\t'. remove tabs and extract first sentence
        
        basename = Path(fname).stem
        lab_name = f"{basename}.lab"

        #fake that each wavfile is a separate speaker
        word_labs = Path(KEYWORD + "/word_labs")
        os.makedirs(word_labs / basename)

        #write transcript
        lab_file = word_labs / basename / lab_name
        with open( lab_file, "w", encoding="utf8") as fh:
            fh.write(transcript)

        shutil.copy2(clips_dir / fname, word_labs / basename) #copy audio to world lab dir
        shutil.copy2(clips_dir / fname, data_and_aligns / "data") #copy audio to data and aligns
        shutil.copy2(lab_file, data_and_aligns / "data") #copy lab to data and aligns
        written += 1
        print(f"labs written: {written}")

        for w in transcript.split(" "):
            all_words_in_src.add( w.lower() ) #cast to lower case to standardize words

    return all_words_in_src

def create_lexicons(lexicon_path, all_words):
    with open(lexicon_path, "w") as fh:
        for w in all_words:
            # filter apostrophes
            w = w.translate(str.maketrans('','',string.punctuation))
            spelling = " ".join(w)
            wl = f"{w}\t{spelling}\n" #separate word from pronunciation using \t as required by mfa
            fh.write(wl)
    return

#go from csv to lexicons
def extract_from_tsv(tsv_path, clips_dir, target_clips, non_target_clips, data_and_aligns, KEYWORD, NUM_OF_CLIPS_PER_CATEGORY):
    df = pd.read_csv(tsv_path, sep='\t') #load data into pandas df

    #separate words into target and non-target
    print("populate t and non_t clips")
    result = populate_t_and_nt_audios(df, clips_dir, target_clips, non_target_clips, KEYWORD, NUM_OF_CLIPS_PER_CATEGORY)

    #CREATE WORD LABS 
    print("creating word labs")
    all_words = set()

    _, target_list = result[0]
    all_target_words = create_word_labs(df, KEYWORD, data_and_aligns, target_list, clips_dir)

    _, non_target_list = result[1]
    all_ntarget_words = create_word_labs(df, KEYWORD, data_and_aligns, non_target_list, clips_dir)

    #CREATE LEXICONS
    print("creating lexicons")
    all_words = all_target_words | all_ntarget_words
    print("Num words", len(all_words))
    lexicon_path = Path(KEYWORD + "/lexicon.txt")
    create_lexicons(lexicon_path, all_words)

    return

def generate_mfas(KEYWORD, NUM_OF_CLIPS_PER_CATEGORY):
    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    clips_dir = Path(base_dir + "cv-corpus-wav/clips/")

    #initialize folders
    target_clips = Path(KEYWORD + '/target/')
    os.makedirs(target_clips)
    non_target_clips = Path(KEYWORD +'/non_target/')
    os.makedirs(non_target_clips)

    data_and_aligns = Path(KEYWORD + "/data_and_alignments/")
    os.makedirs(data_and_aligns / "data")
    os.makedirs(data_and_aligns / "alignments")

    train_tsv = 'cv-corpus-wav/train.tsv'
    extract_from_tsv(train_tsv, clips_dir, target_clips, non_target_clips, data_and_aligns, KEYWORD, NUM_OF_CLIPS_PER_CATEGORY)

    test_tsv = 'cv-corpus-wav/test.tsv'
    extract_from_tsv(test_tsv, clips_dir, target_clips, non_target_clips, data_and_aligns, KEYWORD, NUM_OF_CLIPS_PER_CATEGORY)

    #run mfa train command to produce alignments
    data_path = base_dir + str(data_and_aligns) + "/data/"
    lexicon_path = base_dir + KEYWORD + "/lexicon.txt"
    alignments_path = base_dir + str(data_and_aligns) + "/alignments/"
    mfa_train_command = "mfa train --clean " + data_path + " " + lexicon_path + " " + alignments_path
    os.system(mfa_train_command)

    return 

# mfa train --clean /media/cbanbury/T7/jquaye/abantu/data_and_alignments/data/  /media/cbanbury/T7/jquaye/abantu/lexicon.txt /media/cbanbury/T7/jquaye/abantu/data_and_alignments/alignments