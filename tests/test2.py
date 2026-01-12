import os
import json
import copy

mi_file = 'MI308_beam3_trace_10times_1230_step.json'

nv_file = 'H20_decoder_beam3_trace_10times_cuda_graph_1230_step.json'

def load(f):
    with open(f, 'r') as fp:
        mi_json = json.load(fp)
    
    return mi_json

mi_1 = load(mi_file)
nv_1 = load(nv_file)


def remove_idx_triton(p):
    for e in p['traceEvents']:
        if "triton" in e["name"]:
            idx = e["name"].rindex("_")
            e["name"] = e["name"][:idx]

    return p


def simplify_name(p):
    for e in p['traceEvents']:
        if "<" in e["name"] and ">" in e["name"]:
            idx = e["name"].index("<")
            e["name"] = e["name"][:idx]

    return p

nv_1 = simplify_name(nv_1)
mi_1 = simplify_name(mi_1)

def one_block(profile, split):
    profile_new = copy.deepcopy(profile)
    profile_new['traceEvents'] = []
    start = False
    step = 0
    for e in profile['traceEvents']:
        if e["name"] == split:
            if step == 258:
                break

            if step == 256:
                start = True
            
            step += 1

        if start is False:
            continue
        profile_new['traceEvents'].append(e)

    return profile_new

nv_1 = one_block(nv_1, "triton_tem_fused_addmm_t_transpose_view_26")
mi_1 = one_block(mi_1, "triton_tem_fused_addmm_t_transpose_view_26")


def get_base_time(profile):
    return profile['traceEvents'][0]['ts']

mi_new_base = get_base_time(mi_1)
nv_new_base = get_base_time(nv_1)

def update_base(profile, offset):
    for e in profile['traceEvents']:
        e['ts'] += offset

    return profile

if mi_new_base > nv_new_base:
    nv_1 = update_base(nv_1, mi_new_base - nv_new_base)
else:
    mi_1 = update_base(mi_1, nv_new_base - mi_new_base)


nv_1 = remove_idx_triton(nv_1)
mi_1 = remove_idx_triton(mi_1)

def save(p, pp):
    with open(pp, 'w') as fp:
        json.dump(p, fp)


save(nv_1, "nv_1.json")
save(mi_1, "mi_1.json")

def fuse_op(p, fusions):
    p_new = copy.deepcopy(p)
    p_new["traceEvents"] = []
    for a,b in fusions:
        idx = 0
        while idx < len(p['traceEvents']):
            if p['traceEvents'][idx]["name"] == a and idx != len(p['traceEvents']) - 1:
                if p['traceEvents'][idx+1]["name"] == b:
                    e = copy.deepcopy(p['traceEvents'][idx])
                    e["name"] = f'{a}_{b}'
                    e['dur'] = e['dur'] + p['traceEvents'][idx+1]["dur"]
                    p_new["traceEvents"].append(e)
                    idx += 2
                    continue

            p_new["traceEvents"].append(p['traceEvents'][idx])
            idx += 1
    return p_new

nv_1 = fuse_op(nv_1, [["triton_poi_fused_addmm_relu_view", "nvjet_hsh_8x64_64x16_4x1_v_bz_TNN"],])

def collect(nv, mi):
    nv_new = copy.deepcopy(nv)
    mi_new = copy.deepcopy(mi)
    for idx, e in enumerate(mi_new['traceEvents']):
        if idx >= len(nv_new['traceEvents']):
            break
        
        nv_new['traceEvents'][idx]['ts'] = e['ts']

    return nv_new, mi_new

nv_1, mi_1 = collect(nv_1, mi_1)

def split(profile, is_nv=False):
    pid = 100
    if is_nv:
        pid = 200

    profile_new = copy.deepcopy(profile)
    for e in profile_new['traceEvents']:
        if "triton_" in e["name"]:
            tid = pid + 10
            if "_tem_" in e["name"]:
                tid += 10
        else:
            # others:
            tid = pid + 0
        e['tid'] = tid
        e['pid'] = pid
    return profile_new

nv_1 = split(nv_1, True)
mi_1 = split(mi_1, False)

def merge_nv_mi(nv, mi):
    nv_new = copy.deepcopy(nv)
    for e in mi['traceEvents']:
        nv_new['traceEvents'].append(e)
    return nv_new

merge = merge_nv_mi(nv_1, mi_1)

save(merge, "1L1B.json")


