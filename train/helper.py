def move_data_pair_to(device, data_pair):
    # move data to GPU
    for k in data_pair:
        if "path" in k:
            # do not move string
            continue
        else:
            data_pair[k] = data_pair[k].to(device)
