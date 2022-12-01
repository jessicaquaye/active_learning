import os
import shutil
import sox
import textgrid

from pathlib import Path
    
def trim_and_pad(keyword, src_file, dest_file, tgfile):
    tg = textgrid.TextGrid.fromFile(tgfile)

    for interval in tg[0]:
        if interval.mark != keyword:
            continue
        
        #trim keyword from the audio
        src_len = sox.file_info.duration(src_file)
        start_s = interval.minTime
        end_s = start_s + 1

        if end_s > src_len: #check for trimming beyond end of audio
            end_s = src_len

        tfm = sox.Transformer()
        tfm.convert(samplerate=16000)
        tfm.trim(start_s, end_s)

        cropped_file = 'cropped.wav' #temp holding place after cropping in case padding is necessary
        tfm.build(src_file, cropped_file)

        duration_s = sox.file_info.duration(cropped_file) #pad if necessary
        if duration_s < 1: #need to pad
            pad_amt_s = (1.0 - duration_s) / 2.0
            tfm.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
            tfm.build(cropped_file, dest_file)
            
        else: #copy over 1s audio
            shutil.copy2(Path(cropped_file), Path(dest_file))
        
        os.remove(cropped_file)
    
    return dest_file

def create_one_s(KEYWORD):
    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"

    clips_dir = base_dir + "cv-corpus-wav/clips/"
    one_s_dest = base_dir + KEYWORD + "/one_s/"
    os.makedirs(one_s_dest)

    #for each text grid, extract one_s word
    tg_dir = base_dir + KEYWORD + "/data_and_alignments/alignments/"
    for tgfname in os.listdir(tg_dir):
        fwav = Path(tgfname).stem + ".wav"
        src_file = clips_dir + fwav
        dest_file = one_s_dest + fwav
        tgfile = tg_dir + tgfname

        trim_and_pad(KEYWORD, src_file, dest_file, tgfile)
    
    return


