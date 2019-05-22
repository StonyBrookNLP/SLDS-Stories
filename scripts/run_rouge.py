import os
import sys
import shutil
import pyrouge
import logging

def rouge_eval(ref_dir, dec_dir):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  r = pyrouge.Rouge155('/home/lshekhar/ROUGE-1.5.5')
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING)
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write=None):
  """Log ROUGE results to screen and write to file.
  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  print(log_str) # log to screen


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

def get_sents(line):
    decoded_words = line.split()
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError: # there is text remaining that doesn't end in "."
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
        decoded_words = decoded_words[fst_period_idx+1:] # everything else
        decoded_sents.append(' '.join(sent))
    return decoded_sents


def write_for_rouge(ref_file, output_file, _rouge_ref_dir, _rouge_dec_dir):
    # write temp files
    with open(ref_file) as fr: # has empty lines
        lines = fr.readlines()  
        lines = [line.rstrip('\n') for line in lines]
        lines = [line for line in lines if line]
        len_refs = len(lines)
        #print(lines)
        print("Total REFS: {}".format(len(lines)))
        #exit()
        for idx, line in enumerate(lines): 
            fpath = os.path.join(_rouge_ref_dir, "{:06d}_reference.txt".format(idx))
            # split story into sents
            sents = get_sents(line)
            sents = [make_html_safe(w) for w in sents]
                   
            with open(fpath, 'w') as fw:
                for i,sent in enumerate(sents):
                    fw.write(sent) if i==len(sents)-1 else fw.write(sent+"\n")
            #break

    with open(output_file) as fd:
        lines = fd.readlines()
        print("Total OUTPUT: {}".format(len(lines)))
        lines = [line.rstrip('\n') for line in lines]
        len_out = len(lines)
        for idx, line in enumerate(lines):
            fpath = os.path.join(_rouge_dec_dir, "{:06d}_decoded.txt".format(idx))
            sents = get_sents(line)
            sents = [make_html_safe(w) for w in sents]

            with open(fpath, 'w') as fw:
                for i,sent in enumerate(sents):
                    fw.write(sent) if i==len(sents)-1 else fw.write(sent+"\n")
            #break
    
    assert len_refs == len_out, "Ref and output file must be same in length."
    print("Temp files created under dirs.")


def do_rouge(ref_file, output_file):

    print("!!SPLITS STORY INTO SENTENCES USING . ONLY!!")

    # parent dirs to keep the temp files in
    _rouge_ref_dir = "./rg_temp_ref"
    _rouge_dec_dir = "./rg_temp_out"

    #delete_temp_dirs
    if os.path.exists(_rouge_ref_dir):
        shutil.rmtree(_rouge_ref_dir)
    if os.path.exists(_rouge_dec_dir):
        shutil.rmtree(_rouge_dec_dir)

    #create temp dirs
    os.mkdir(_rouge_ref_dir)
    os.mkdir(_rouge_dec_dir)
    print("Temp dirs {} and {} created.".format(_rouge_ref_dir, _rouge_dec_dir))

    write_for_rouge(ref_file, output_file, _rouge_ref_dir, _rouge_dec_dir)

    # calculate rouge
    results_dict = rouge_eval(_rouge_ref_dir, _rouge_dec_dir)
    rouge_log(results_dict)

    #delete_temp_dirs
   
    shutil.rmtree(_rouge_ref_dir) 
    shutil.rmtree(_rouge_dec_dir)
    print("Temp dirs {} and {} deleted".format(_rouge_ref_dir, _rouge_dec_dir))


# arg1 - ref files; arg2 - output file
#print("ARG1: reference file. ARG2: output file")
#do_rouge(sys.argv[1], sys.argv[2])
