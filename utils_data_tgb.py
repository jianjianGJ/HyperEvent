import os
from tqdm import tqdm
import torch
'''
Notice: -1 is uniformly used to represent an empty neighbor placeholder.
'''
def distance_u_v(a,b):
    if a.dim() == 4:# 1-hop
        overlap = a.unsqueeze(-2) == b.unsqueeze(-1)
        neg_mask = (a.unsqueeze(-2)<0) | (b.unsqueeze(-1)<0)
        overlap[neg_mask] = 0 #-1 match -1 not ok
        return overlap.float().mean(dim=(-1, -2))
    elif a.dim() == 3:#0-hop
        overlap = a.unsqueeze(-1) == b
        neg_mask = a<0
        overlap = overlap.float().mean(-1)
        overlap[neg_mask] = 0
        return overlap
    else:
        raise ValueError()

def get_segment(dataset, phase, num_segment_train, num_segment_val,
                                   num_segment_test, num_neighbor, path='./adjtables'):
    '''get src,dst,ts  after segmentation
    new shape:[num_segment, ceil(region_length/num_segment)]'''
    src, dst, ts = dataset.src, dataset.dst, dataset.ts
    id_train_last = torch.arange(dataset.num_edges)[dataset.train_mask][-1].item()
    id_val_last = torch.arange(dataset.num_edges)[dataset.val_mask][-1].item()
    if phase == 'train':
        start, end = 0, id_train_last
        num_segment = num_segment_train
    elif phase == 'val':
        start, end = id_train_last+1, id_val_last
        num_segment = num_segment_val
    elif phase == 'test':
        start, end = id_val_last+1, dataset.num_edges - 1
        num_segment = num_segment_test
    else:
        raise ValueError('Wrong phase')
    region_length = end - start + 1
    len_seg = (region_length + num_segment - 1) // num_segment
    src_seg = torch.full((num_segment, len_seg), -1, dtype=src.dtype)
    dst_seg = torch.full((num_segment, len_seg), -1, dtype=dst.dtype)
    ts_seg = torch.full((num_segment, len_seg), -1, dtype=ts.dtype)
    valid_data_src = src[start: end+1].contiguous()
    valid_data_dst = dst[start: end+1].contiguous()
    valid_data_ts = ts[start: end+1].contiguous()
    
    flat_src = src_seg.view(-1)
    flat_dst = dst_seg.view(-1)
    flat_ts = ts_seg.view(-1)
    flat_src[:region_length] = valid_data_src
    flat_dst[:region_length] = valid_data_dst
    flat_ts[:region_length] = valid_data_ts
    # load adjtable precalculated
    precalculate_adjtable_for_segments(dataset, num_segment_train, num_segment_val,
                                        num_segment_test, num_neighbor, path=path)
    save_path = os.path.join(path, f"{dataset.name}_adjtable_seg_{phase}_{num_segment}_{num_neighbor}.pt")
    adjtable_seg =  torch.load(save_path)
    return src_seg, dst_seg, ts_seg, adjtable_seg
def get_batch(src_seg, dst_seg, ts_seg, i):
    '''one batch contains num_segment samples'''
    src_i, dst_i, ts_i = src_seg[:,i], dst_seg[:,i], ts_seg[:,i]
    valid_mask = src_i > -1
    src_i, dst_i, ts_i = src_i[valid_mask], dst_i[valid_mask], ts_i[valid_mask]
    return src_i.unsqueeze(-1), dst_i.unsqueeze(-1), ts_i.unsqueeze(-1)
def load_ns_eval(dataset, phase, src_i=None, dst_i=None, ts_i=None):
    dst_seg_neg = dataset.negative_sampler.query_batch(src_i.squeeze(-1), dst_i.squeeze(-1), ts_i.squeeze(-1), split_mode=phase)
    length = [len(i) for i in dst_seg_neg]
    length_min = min(length)
    # fix n_neg for all test samples for tensor calculation, add -1 if not enough
    # notice that it can not improve mrr when add more negative samples
    if dataset.name == 'tgbl-wiki': 
        n_neg = 999 # max ns sample in tgbl-wiki
    elif dataset.name == 'tgbl-review':
        n_neg = 100 # max ns sample in tgbl-review
    else:
        n_neg = 20
    if length_min == n_neg:
        dst_seg_neg = torch.tensor(dst_seg_neg, dtype=dataset.src.dtype, device=src_i.device)
        return dst_seg_neg
    else:
        for q in dst_seg_neg:
            q += [-1] * (n_neg - len(q)) 
        dst_seg_neg = torch.tensor(dst_seg_neg, dtype=dataset.src.dtype, device=src_i.device)
        return dst_seg_neg
def precalculate_adjtable_for_segments(dataset, num_segment_train, num_segment_val,
                                   num_segment_test, num_neighbor, path='./adjtables'):
    os.makedirs(path, exist_ok=True)
    save_path_train = os.path.join(path, f"{dataset.name}_adjtable_seg_train_{num_segment_train}_{num_neighbor}.pt")
    save_path_val = os.path.join(path, f"{dataset.name}_adjtable_seg_val_{num_segment_val}_{num_neighbor}.pt")
    save_path_test = os.path.join(path, f"{dataset.name}_adjtable_seg_test_{num_segment_test}_{num_neighbor}.pt")
    if os.path.exists(save_path_train) and os.path.exists(save_path_val) and os.path.exists(save_path_test):
        print('adjtables has exist!!!')
        return 
    num_nodes = max(dataset.src.max().item(), dataset.dst.max().item()) + 1
    id_train_last = torch.arange(dataset.num_edges)[dataset.train_mask][-1].item()
    id_val_last = torch.arange(dataset.num_edges)[dataset.val_mask][-1].item()
    seg_set_point_train = []
    seg_set_point_val = []
    seg_set_point_test = []
    for phase in ['train', 'val', 'test']:# get segment points
        if phase == 'train':
            start, end = 0, id_train_last
            num_segment = num_segment_train
            seg_set_point = seg_set_point_train
        elif phase == 'val':
            start, end = id_train_last+1, id_val_last
            num_segment = num_segment_val
            seg_set_point = seg_set_point_val
        elif phase == 'test':
            start, end = id_val_last+1, dataset.num_edges - 1
            num_segment = num_segment_test
            seg_set_point = seg_set_point_test
        else:
            raise ValueError('Wrong phase')
        region_length = end - start + 1
        len_seg = (region_length + num_segment - 1) // num_segment
        seg_set_point += [start+len_seg*i for i in range(num_segment)]
    adjtable_seg_train = []
    adjtable_seg_val = []
    adjtable_seg_test = []
    adjtable = torch.full((num_nodes+1, num_neighbor), -1)
    # save adjtable at segment points during adjtable updates
    for i in tqdm(range(dataset.num_edges), desc='adjtable'):
        if i in seg_set_point_train: #Notice: save adjtable befor update, so no leakage
            adjtable_seg_train.append(adjtable.clone())
            if i == seg_set_point_train[-1]:
                adjtable_seg_train = torch.stack(adjtable_seg_train)
                torch.save(adjtable_seg_train, save_path_train)
                del adjtable_seg_train
        if i in seg_set_point_val:
            adjtable_seg_val.append(adjtable.clone())
            if i == seg_set_point_val[-1]:
                adjtable_seg_val = torch.stack(adjtable_seg_val)
                torch.save(adjtable_seg_val, save_path_val)
                del adjtable_seg_val
        if i in seg_set_point_test:
            adjtable_seg_test.append(adjtable.clone())
            if i == seg_set_point_test[-1]:
                adjtable_seg_test = torch.stack(adjtable_seg_test)
                torch.save(adjtable_seg_test, save_path_test)
                del adjtable_seg_test
        src, dst = dataset.src[i].item(), dataset.dst[i].item()
        adjtable[src][:-1] = adjtable[src][1:].clone() #update adjtable
        adjtable[src][-1] = dst
        adjtable[dst][:-1] = adjtable[dst][1:].clone()
        adjtable[dst][-1] = src

# Section: Efficient Implementation (GPU tensor computation instead of CPU multi-thread)
# We track the shape of tensors for code readability
def get_x_y(adjtable_seg, src_i, dst_i, query_pos_first, num_latest, big_hood):
    len_query = query_pos_first.size(1) # positive sample in the first position in query; len_query=1+num_negative_samples
    # src_i(num_segment,1) query_pos_first(num_segment, len_query) 
    src_dst = torch.cat([src_i.repeat(1,len_query).unsqueeze(-1), query_pos_first.unsqueeze(-1)], dim=-1).view(-1,2)
    # src_dst(num_segment*len_query, 2)
    seg_dim_arange = torch.arange(src_i.size(0), device=src_i.device).unsqueeze(-1)
    src_latest = adjtable_seg[seg_dim_arange,src_i][:,:,-num_latest:]
    dst_latest = adjtable_seg[seg_dim_arange,query_pos_first][:,:,-num_latest:]
    #  src_latest(num_segment, 1, num_latest)  dst_latest(num_segment, len_query, num_latest) 
    src_dst_latest = torch.cat([src_latest.repeat(1,len_query,1), dst_latest], dim=-1).view(-1,2*num_latest)
    # src_dst_latest(num_segment*len_query, 2*num_latest)
    i = src_dst.unsqueeze(1).repeat(1, 2*num_latest, 2)
    # i(num_segment*len_query, 2*num_latest, 4) correlation vector dimention:4
    j = src_dst_latest.unsqueeze(-1).repeat(1, 1, 4)
    j[:,:num_latest,:2] = i[:,:num_latest,::2]
    j[:,num_latest:,2:] = i[:,num_latest:,1::2]
    i = i.view(-1, len_query,2*num_latest, 4) #torch.Size([num_segment, len_query, 2*num_latest, 4])
    j = j.view(-1, len_query,2*num_latest, 4) #torch.Size([num_segment, len_query, 2*num_latest, 4])
    # adjtable_seg(num_segment, num_nodes, num_neighbor)
    i_neighbor = adjtable_seg[seg_dim_arange.unsqueeze(-1).unsqueeze(-1), i] #torch.Size([num_segment, len_query, 2*num_latest, 4, num_neighbor])
    if big_hood:
        num_latest_2hop = int(pow(adjtable_seg.size(2), 0.5))
        i_neighbor_longer = adjtable_seg[seg_dim_arange.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), i_neighbor[:,:,:,:,-num_latest_2hop:]][:,:,:,:,:,-num_latest_2hop:].contiguous()
    i_neighbor = i_neighbor.view(-1, i_neighbor.size(2), i_neighbor.size(3), i_neighbor.size(4)) #torch.Size([num_segment*len_query, 2*num_latest, 4, num_neighbor])
    if big_hood:    
        i_neighbor_longer = i_neighbor_longer.view(-1, i_neighbor_longer.size(2),i_neighbor_longer.size(3),i_neighbor_longer.size(4)*i_neighbor_longer.size(5))
    j_neighbor = adjtable_seg[seg_dim_arange.unsqueeze(-1).unsqueeze(-1), j] #torch.Size([num_segment, len_query, 2*num_latest, 4, num_neighbor])
    if big_hood:
        j_neighbor_longer = adjtable_seg[seg_dim_arange.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), j_neighbor[:,:,:,:,-num_latest_2hop:]][:,:,:,:,:,-num_latest_2hop:].contiguous()
    j_neighbor = j_neighbor.view(-1, j_neighbor.size(2), j_neighbor.size(3), j_neighbor.size(4)) #torch.Size([num_segment*len_query, 2*num_latest, 4, num_neighbor])
    if big_hood:
        j_neighbor_longer = j_neighbor_longer.view(-1, j_neighbor_longer.size(2),j_neighbor_longer.size(3),j_neighbor_longer.size(4)*j_neighbor_longer.size(5))
    
    x_two_hop = distance_u_v(i_neighbor, j_neighbor)# torch.Size([num_segment*len_query, 2*num_latest, 4)
    if big_hood: # use 0-hop, 1-hop, 2-hop distance
        x_one_hop = distance_u_v(i.view(-1, 2*num_latest, 4), j_neighbor)
        x_three_hop = distance_u_v(i_neighbor_longer, j_neighbor_longer)
        x = torch.cat([x_one_hop, x_two_hop, x_three_hop], dim=-1)
    else:
        x = x_two_hop
    y = torch.zeros((x.shape[0],1), device=x.device)
    y[::len_query] = 1 # positive sample in thr first position
    
    # update adjtable_seg
    seg_ids = seg_dim_arange.view(-1)          
    src_flat = src_i.view(-1)                  # [num_segment]
    dst_flat = dst_i.view(-1)                  # [num_segment]
    current_src_adjtable = adjtable_seg[seg_ids, src_flat]  # [num_segment, num_neighbor]
    updated_src_adjtable = torch.roll(current_src_adjtable, shifts=-1, dims=-1)
    updated_src_adjtable[:, -1] = dst_flat
    adjtable_seg[seg_ids, src_flat] = updated_src_adjtable 

    current_dst_adjtable = adjtable_seg[seg_ids, dst_flat]  # [num_segment, num_neighbor]
    updated_dst_adjtable = torch.roll(current_dst_adjtable, shifts=-1, dims=-1)
    updated_dst_adjtable[:, -1] = src_flat
    adjtable_seg[seg_ids, dst_flat] = updated_dst_adjtable  
    return x, y
