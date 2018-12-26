import os

def safeMakeDir(tdir):
    if not os.path.isdir(tdir):
        os.mkdir(tdir)

def safeMakeDirs(tdir):
    if not os.path.isdir(tdir):
        os.makedirs(tdir)
