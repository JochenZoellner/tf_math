import errno
import os
import shutil


def cp_copy(src, dest, exist_ok=True):
    """create path copy"""
    if not os.path.isdir(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if not exist_ok:
        if os.path.isfile(dest) or os.path.isdir(dest):
            raise IOError("File already exist! {}".format(dest))
    shutil.copy(src, dest)


def file_path_with_mkdirs(path_with_filen_name):
    """will create all dirs to save the file to the given path"""
    if not os.path.isdir(os.path.dirname(path_with_filen_name)):
        try:
            os.makedirs(os.path.dirname(path_with_filen_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return path_with_filen_name
