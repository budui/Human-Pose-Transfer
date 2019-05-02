import torch


def move_data_pair_to(device, data_pair):
    # move data to GPU
    for k in data_pair:
        if "path" in k:
            # do not move string
            continue
        else:
            data_pair[k] = data_pair[k].to(device)


def load_model(model_class, model_save_path, device):
    model = model_class()
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    return model
