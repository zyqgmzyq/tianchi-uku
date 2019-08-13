from . import *
import os

def logging(filename, message):
    with open(filename, 'a+') as f:
        if not message.endswith('\n'):
            message += '\n'
        f.write(message)

def mkdirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)