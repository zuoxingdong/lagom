import time
import torch

from lagom.utils import color_str


def checkpointer(mode, config, obj=[], state_obj=[]):
    assert mode in ['save', 'load']
    t0 = time.perf_counter()
    if mode == 'save':
        state_dicts = [x.state_dict() for x in state_obj]
        import pickle
        torch.save(obj=[obj, state_dicts], f=config.resume_checkpointer, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        msg = f'$ Checkpoint saved at {config.resume_checkpointer.as_posix()}. ({round(time.perf_counter() - t0, 3)}s)'
        print(color_str(msg, color='yellow', bold=True))
    elif mode == 'load':
        obj, state_dicts = torch.load(f=config.resume_checkpointer, map_location=config.device)
        for x, x_dict in zip(state_obj, state_dicts):
            x.load_state_dict(x_dict)
        msg = f'$ Checkpoint loaded from {config.resume_checkpointer.as_posix()}. ({round(time.perf_counter() - t0, 3)}s)'
        print(color_str(msg, color='yellow', bold=True))
        return obj
    else:
        raise NotImplementedError
