def load_cache(ckpt_file, cache_file, force_update):
    if cache_file.exists():
        if force_update is True:
            return False
        elif force_update is False:
            return True
        else:
            return cache_file.stat().st_mtime > ckpt_file.stat().st_mtime
    else:
        return False
