import os
import pandas as pd
import shutil
import string
from pathlib import Path

def populate_t_and_nt_audios(df, clips_dir, target_clips, non_target_clips, keyword, num_of_clips_per_category):
    #extract target and non-target
    files_w_key = df[ df['sentence'].str.contains(keyword).fillna(False) ]
    files_wout_key = df[ ~df['sentence'].str.contains(keyword).fillna(False) ]

    #select target and non-target that will be used for training
    target = list(files_w_key['path'])[:num_of_clips_per_category]
    non_target = list(files_wout_key['path'])[:num_of_clips_per_category]

    print("target clips size: ", len(target))
    print("non target clips size: ", len(non_target))
    
    #copy audio clips into respective folders
    print("iterating through target files")
    for fname in target:
        if os.path.exists(clips_dir / fname):
            shutil.copy2(clips_dir / fname, target_clips / fname)
    print("iterating through non_target files")
    for fname in non_target:
        if os.path.exists(clips_dir / fname):
            shutil.copy2(clips_dir / fname, non_target_clips / fname)
    return [(target_clips, target), (non_target_clips, non_target)]

def create_word_labs(df, keyword, data_and_aligns, clips_of_interest_list, clips_dir):
    written = 0

    all_words_in_src = set()

    for fname in clips_of_interest_list:
        row = df.loc[(df['path'] == fname)]
        sentence = row['sentence'].values[0]
        transcript = sentence.split('\t')[0] #some of the sentences contained '\t'. remove tabs and extract first sentence
        
        basename = Path(fname).stem
        lab_name = f"{basename}.lab"

        #fake that each wavfile is a separate speaker
        word_labs = Path(keyword + "/word_labs")
        os.makedirs(word_labs / basename)

        #write transcript
        lab_file = word_labs / basename / lab_name
        with open(lab_file, "w", encoding="utf8") as fh:
            fh.write(transcript)

        if os.path.exists(clips_dir / fname):
            # shutil.copy2(clips_dir / fname, word_labs / basename) #copy audio to world lab dir
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
def extract_from_tsv(tsv_path, clips_dir, target_clips, non_target_clips, data_and_aligns, keyword, num_of_clips_per_category):
    df = pd.read_csv(tsv_path, sep='\t') #load data into pandas df

    #separate words into target and non-target
    print("populate t and non_t clips")
    result = populate_t_and_nt_audios(df, clips_dir, target_clips, non_target_clips, keyword, num_of_clips_per_category)

    #CREATE WORD LABS 
    print("creating word labs")
    all_words = set()

    _, target_list = result[0]
    print("target list from pop t: ", len(target_list))
    all_target_words = create_word_labs(df, keyword, data_and_aligns, target_list, clips_dir)


    print("starting non-target processing")
    _, non_target_list = result[1]
    print("non target list from pop nt: ", len(non_target_list))
    all_ntarget_words = create_word_labs(df, keyword, data_and_aligns, non_target_list, clips_dir)

    #CREATE LEXICONS
    print("creating lexicons")
    all_words = all_target_words | all_ntarget_words
    print("Num words", len(all_words))
    lexicon_path = Path(keyword + "/lexicon.txt")
    create_lexicons(lexicon_path, all_words)

    return

def generate_mfas(keyword, num_of_clips_per_category):
    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"
    clips_dir = Path(base_dir + "cv-corpus-wav/clips/")

    # initialize folders
    non_target_clips = Path(keyword +'/non_target/')
    os.makedirs(non_target_clips)

    data_and_aligns = Path(keyword + "/data_and_alignments/")
    os.makedirs(data_and_aligns / "data")
    os.makedirs(data_and_aligns / "alignments")

    print("extracting training clips")
    train_tsv = 'cv-corpus-wav/train.tsv'
    train_target_clips = Path(keyword + '/target/train/')
    os.makedirs(train_target_clips)
    extract_from_tsv(keyword=keyword,
                    tsv_path=train_tsv,
                    clips_dir=clips_dir,
                    target_clips=train_target_clips,
                    non_target_clips=non_target_clips,
                    data_and_aligns=data_and_aligns,
                    num_of_clips_per_category=num_of_clips_per_category)

    print("extracting validation clips")
    dev_tsv = 'cv-corpus-wav/dev.tsv'
    dev_target_clips = Path(keyword + '/target/dev/')
    os.makedirs(dev_target_clips)
    extract_from_tsv(keyword=keyword,
                    tsv_path=dev_tsv,
                    clips_dir=clips_dir,
                    target_clips=dev_target_clips,
                    non_target_clips=non_target_clips,
                    data_and_aligns=data_and_aligns,     
                    num_of_clips_per_category=num_of_clips_per_category)

    print("extracting testing clips")
    test_tsv = 'cv-corpus-wav/test.tsv'
    test_target_clips = Path(keyword + '/target/test/')
    os.makedirs(test_target_clips)
    extract_from_tsv(keyword=keyword,
                    tsv_path=test_tsv,
                    clips_dir=clips_dir,
                    target_clips=test_target_clips,
                    non_target_clips=non_target_clips,
                    data_and_aligns=data_and_aligns,
                    num_of_clips_per_category=num_of_clips_per_category)

    #run mfa train command to produce alignments
    data_path = base_dir + str(data_and_aligns) + "/data/"
    lexicon_path = base_dir + keyword + "/lexicon.txt"
    alignments_path = base_dir + str(data_and_aligns) + "/alignments/"
    mfa_train_command = "mfa train --clean " + data_path + " " + lexicon_path + " " + alignments_path
    os.system(mfa_train_command)

    return 

# mfa train --clean /media/cbanbury/T7/jquaye/abantu/data_and_alignments/data/  /media/cbanbury/T7/jquaye/abantu/lexicon.txt /media/cbanbury/T7/jquaye/abantu/data_and_alignments/alignments