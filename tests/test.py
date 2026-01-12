import os
import json
import copy

mi_file = 'MI308_beam3_trace_10times_1230.json'
nv_file = 'H20_decoder_beam3_trace_10times_cuda_graph_1230.json'
mi_file_new = 'MI308_beam3_trace_10times_1230_step.json'

nv_file_new = 'H20_decoder_beam3_trace_10times_cuda_graph_1230_step.json'

def remove_host(file):

    with open(file, 'r') as fp:
        mi_json = json.load(fp)

    mi_json_new = copy.deepcopy(mi_json)

    mi_json_new['traceEvents'] = []

    for e in mi_json['traceEvents']:
        #"cat": "kernel"
        if 'cat' in e and e['cat'] == 'kernel':
            mi_json_new['traceEvents'].append(e)
    return mi_json_new

mi_new = remove_host(mi_file)
nv_new = remove_host(nv_file)


def one_step(profile, split='triton_poi_fused_add_embedding_eq_expand_index_view_where_0'):
    profile_new = copy.deepcopy(profile)
    profile_new['traceEvents'] = []
    start = False
    step = 0
    for e in profile['traceEvents']:
        if start is True and e["name"] == split:
            break

        if start is False and e["name"] == split:
            if step == 2:
                start = True
            else:
                step += 1

        if start is False:
            continue
        del e['args']
        profile_new['traceEvents'].append(e)

    return profile_new

mi_new = one_step(mi_new, 'triton_poi_fused_add_embedding_eq_expand_index_view_where_0')
nv_new = one_step(nv_new, 'triton_poi_fused_add_embedding_eq_expand_index_view_where_0')


def get_base_time(profile):
    return profile['traceEvents'][0]['ts']

mi_new_base = get_base_time(mi_new)
nv_new_base = get_base_time(nv_new)

def update_base(profile, offset):
    for e in profile['traceEvents']:
        e['ts'] += offset

    return profile

if mi_new_base > nv_new_base:
    nv_new = update_base(nv_new, mi_new_base - nv_new_base)
else:
    mi_new = update_base(mi_new, nv_new_base - mi_new_base)


## gen gemm profile
def split_gemm(profile):
    profile_new = copy.deepcopy(profile)
    gemm_tid = 11
    non_gemm_tid = 12
    for e in profile_new['traceEvents']:
        if '_tem_' in e["name"]:
            e['tid'] = gemm_tid
        else:
            e['tid'] = non_gemm_tid
    return profile_new

mi_gemm = split_gemm(mi_new)
nv_gemm = split_gemm(nv_new)


with open(mi_file_new, 'w') as fp:
    print(mi_file_new, len(mi_new['traceEvents']))
    json.dump(mi_new, fp)

with open(nv_file_new, 'w') as fp:
    print(nv_file_new, len(nv_new['traceEvents']))
    json.dump(nv_new, fp)


# def merge_nv_mi(nv, mi):
#     nv_new = copy.deepcopy(nv)
#     for e in mi['traceEvents']:
#         nv_new['traceEvents'].append(e)
#     return nv_new

# def collect_same_kernel(nv2, mi2, tid=20):
#     mi = copy.deepcopy(mi2)
#     nv = copy.deepcopy(nv2)

#     mi_set = set()
#     nv_set = set()

#     for e in nv['traceEvents']:
#         nv_set.add(e['name'])
        
    
#     for e in mi['traceEvents']:
#         mi_set.add(e['name'])
    
#     same = mi_set.intersection(nv_set)
#     print(len(nv_set), len(mi_set), len(same))

#     for e in nv['traceEvents']:
#         if e["name"] in same:
#             e['tid'] = tid

#     for e in mi['traceEvents']:
#         if e["name"] in same:
#             e['tid'] = tid
    
#     return nv, mi

# nv22, mi22 = collect_same_kernel(nv_new, mi_new)
# merge22 = merge_nv_mi(nv22, mi22)

# with open('merge22.json', 'w') as fp:
#     json.dump(merge22, fp)


# merge = merge_nv_mi(nv_gemm, mi_gemm)

# with open('merge.json', 'w') as fp:
#     json.dump(merge, fp)


# def collect(nv, mi):
#     nv_new = copy.deepcopy(nv)
#     mi_new = copy.deepcopy(mi)
#     for idx, e in enumerate(mi_new['traceEvents']):
#         if idx >= len(nv_new['traceEvents']):
#             break
        
#         nv_new['traceEvents'][idx]['ts'] = e['ts']

#     return nv_new, mi_new

# nv_new, mi_new = collect(nv_new, mi_new)

# merge = merge_nv_mi(nv_new, mi_new)

# with open('merge_align.json', 'w') as fp:
#     json.dump(merge, fp)

