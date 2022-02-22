import os
import sys
import shutil

def detectAndCopy(src, dst):
    assert os.path.isdir(src)

    if os.path.exists(dst):
        print("Destination already exists:", dst)
    else:
        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
        print("Copying from %s to %s" % (src, dst))
        shutil.copytree(src, dst)
        print("Copy done")
    pass

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python file_copy.py src dst")
        exit(-1)
    src = sys.argv[1]
    dst = sys.argv[2]
    detectAndCopy(src, dst)