import torch


def checkpointer(mode, config, **kwargs):
    assert mode in ['save', 'load']
    if mode == 'save':
        data = {key: value.state_dict() if hasattr(value, 'state_dict') else value for key, value in kwargs.items()}
        import pickle
        torch.save(obj=data, f=config.resume_checkpointer, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        print(f'$ Checkpoint saved at {config.resume_checkpointer.as_posix()}.')
    elif mode == 'load':
        data = torch.load(f=config.resume_checkpointer, map_location=config.device)
        print(f'$ Checkpoint loaded from {config.resume_checkpointer.as_posix()}.')
        out = {}
        for key, value in kwargs.items():
            if hasattr(value, 'load_state_dict'):
                value.load_state_dict(data[key])
            else:
                out[key] = data[key]
        return out
    else:
        raise NotImplementedError
