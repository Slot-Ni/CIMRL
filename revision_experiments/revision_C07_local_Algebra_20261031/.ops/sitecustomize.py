"""Windows checkpoint I/O hotfix; no numerical or experiment-plan changes.
Loaded by Python via PYTHONPATH; child experiment processes inherit it.
"""
import os
import time

_original_replace = getattr(os.replace, '_ins_original_replace', os.replace)


def checkpoint_replace(source, destination, *args, **kwargs):
    is_checkpoint = (str(source).endswith('_training_latest.tmp') and
                     str(destination).endswith('_training_latest.pt'))
    if not is_checkpoint:
        return _original_replace(source, destination, *args, **kwargs)
    for attempt in range(40):
        try:
            return _original_replace(source, destination, *args, **kwargs)
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(.05 * (attempt + 1), .5))


checkpoint_replace._ins_original_replace = _original_replace
checkpoint_replace._ins_checkpoint_retry = True
os.replace = checkpoint_replace
