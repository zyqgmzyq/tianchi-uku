YUV4MPEG2 (.y4m) Reader/Writer


tested with python-2.7, python-3.3 and python3.4

[![Build Status](https://travis-ci.org/ticapix/python-y4m.svg?branch=master)](https://travis-ci.org/ticapix/python-y4m)


```python
import sys
import y4m


def process_frame(frame):
    # do something with the frame
    pass


if __name__ == '__main__':

    parser = y4m.Reader(process_frame, verbose=True)
    # simulate chunk of data
    infd = sys.stdin
    if sys.hexversion >= 0x03000000: # Python >= 3
        infd = sys.stdin.buffer

    with infd as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            parser.decode(data)
```
