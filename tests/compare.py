import os
import json
import copy

mi_file_new = 'xrec2_bs3_long_kv_opt_triton_opt_original_timeline_tracing_mi308x_new.json'
nv_file_new = 'h20_xrec2_bs3_new.json'

def load(file):

    with open(file, 'r') as fp:
        mi_json = json.load(fp)
    return mi_json


mi_json = load(mi_file_new)
nv_json = load(nv_file_new)
