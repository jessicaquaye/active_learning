import csv
import fire
import glob
import json
import multiprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #suppress TF warnings
import shutil
import shlex
import subprocess
import sys
sys.path.append("/media/cbanbury/T7/jquaye/multilingual_kws/")
import tempfile

from multilingual_kws.embedding import input_data
from multilingual_kws.embedding import batch_streaming_analysis as sa
from multilingual_kws.embedding import transfer_learning
from multilingual_kws.embedding.tpr_fpr import tpr_fpr, get_groundtruth
from pathlib import Path
from typing import Optional, List, Dict

abs_path = os.path.abspath(__file__)
base_dir = os.path.dirname(abs_path) + "/"

def eval(streamtarget: sa.StreamTarget, results: Dict):
    results.update(sa.eval_stream_test(streamtarget))

def inference(
    keywords: List[str],
    iteration_num: int,
    groundtruth: Optional[os.PathLike] = None,
    transcript: Optional[os.PathLike] = None,
    visualizer: bool = False,
    serve_port: int = 8080,
    detection_threshold: float = 0.9,
    inference_chunk_len_seconds: int = 1200,
    language: str = "unspecified_language",
    overwrite: bool = False
):
    """
    Runs inference on a streaming audio file. Example invocation:
      $ python -m embedding.run_inference --keyword mask --modelpath mask_model --wav mask_radio.wav
    Args
      keyword: target keywords for few-shot KWS (pass in as [word1, word2, word3])
      groundtruth: optional path to a groundtruth audio file
      transcript: optional path to a groundtruth transcript (for data visualization)
      visualizer: run the visualization server after performing inference
      serve_port: browser port to run visualization server on
      detection_threshold: confidence threshold for inference (default=0.9)
      inference_chunk_len_seconds: we chunk the wavfile into portions
        to avoid exhausting GPU memory - this sets the chunksize.
        default = 1200 seconds (i.e., 20 minutes)
      language: target language (for data visualization)
      overwrite: preserves (and overwrites) visualization outputs
    """

    abs_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(abs_path) + "/"

    #TODO @jquaye: change name from keywords to KEYWORD because it's just 1
    #set up file paths for streaming, modoel and detections
    streaming_wav = base_dir + keywords + "/v_" + str(iteration_num) + "/streaming/" + keywords + "_stream.wav"       #wav: path to the audio file to analyze
    modelpaths = base_dir + keywords + '/v_' + str(iteration_num) + "/" + keywords + '_5shot'  #modelpaths: comma-demlimited list of paths to finetuned few-shot models
    detections_file = base_dir + keywords + '/v_' + str(iteration_num) + '/detections.json' #write_detections: path to save detections.json

    if len(keywords[0]) == 1:
        print(f"NOTE - assuming a single keyword was passed in: {keywords}")
        keywords = [keywords]
    print(f"Target keywords: {keywords}")

    modelpaths = modelpaths.split(",")
    assert len(modelpaths) == len(
        set(keywords)
    ), f"discrepancy: {len(modelpaths)} modelpaths provided for {len(set(keywords))} keywords"

    # create groundtruth if needed
    if groundtruth is None:
        fd, groundtruth = tempfile.mkstemp(prefix="empty_", suffix=".txt")
        os.close(fd)
        print(f"created {groundtruth}")
        created_temp_gt = True
    else:
        created_temp_gt = False

    for p in modelpaths:
        assert os.path.exists(p), f"{p} inference model not found"
    assert os.path.exists(streaming_wav), f"{streaming_wav} streaming audio wavfile not found"
    assert Path(streaming_wav).suffix == ".wav", f"{streaming_wav} filetype not supported"
    assert (
        inference_chunk_len_seconds > 0
    ), "inference_chunk_len_seconds must be positive"

    print(f"performing inference using detection threshold {detection_threshold}")

    unsorted_detections = []
    for keyword, modelpath in zip(keywords, modelpaths):
        flags = sa.StreamFlags(
            wav=streaming_wav,
            ground_truth=groundtruth,
            target_keyword=keyword,
            detection_thresholds=[detection_threshold],
            average_window_duration_ms=100,
            suppression_ms=500,
            time_tolerance_ms=750,  # only used when graphing
            max_chunk_length_sec=inference_chunk_len_seconds,
        )
        streamtarget = sa.StreamTarget(
            target_lang=language,
            target_word=keyword,
            model_path=modelpath,
            stream_flags=[flags],
            destination_result_inferences= base_dir + keyword + '/v_' + str(iteration_num) + '/inferences.txt'
        )
        # manager = multiprocessing.Manager()
        # results = manager.dict()
        # # TODO(mmaz): note that the summary tpr/fpr calculated within eval is incorrect when multiple
        # # targets are being evaluated - groundtruth_labels.txt contains multiple targets but
        # # each model is only single-target (at the moment)
        # p = multiprocessing.Process(target=eval, args=(streamtarget, results))
        # p.start()
        # p.join()

        results = {}
        eval(streamtarget, results)

        unsorted_detections.extend(results[keyword][0][1][detection_threshold][1])

    detections_with_confidence = list(sorted(unsorted_detections, key=lambda d: d[1]))

    # for d in detections_with_confidence:
    #     print(d)

    # cleanup groundtruth if needed
    if created_temp_gt:
        os.remove(groundtruth)
        print(f"deleted {groundtruth}")
        # no groundtruth
        detections_with_confidence = [
            dict(keyword=d[0], time_ms=d[1], confidence=d[2], groundtruth="ng")
            for d in detections_with_confidence
        ]
    else:
        # modify detections using groundtruth
        groundtruth_data = []
        with open(groundtruth, "r") as fh:
            reader = csv.reader(fh)
            for row in reader:
                groundtruth_data.append((row[0], float(row[1])))

        detections_with_confidence = get_groundtruth(
            detections_with_confidence, keywords, groundtruth_data
        )

    detections = dict(
        keywords=keywords,
        detections=detections_with_confidence,
        min_threshold=detection_threshold,
    )

    # # write detections to .txt
    # det_content = [] #initialize as empty but populate if file exists
    # if os.path.exists(detections_file):
    #     det_content = eval(open("detections_file", "r").read())

    # updated_detections = det_content + detections_with_confidence

    with open(detections_file, "w") as det_f:
        json.dump(detections, det_f)
        
    return