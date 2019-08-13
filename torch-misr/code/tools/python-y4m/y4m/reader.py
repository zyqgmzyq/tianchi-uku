
__all__ = ['Reader']

from .frame import Frame


class Reader(object):
    def __init__(self,sourcePath,dstPath,verbose=False):
        self._stream_headers = None
        self._data = bytes()
        self._count = 0
        self._verbose = verbose
        self.dstPath=dstPath
        self.sourcePath=sourcePath

    def _callback(self,frame):
        frame_id=str(frame.frame_id)
        while(len(frame_id)<4):
            frame_id='0'+frame_id
        name=self.sourcePath.split('/')[-1]
        name=name.replace('.y4m','_'+frame_id+'.pickle')
        frame.save(self.dstPath+'/'+name)
        #print('saved yuv as '+self.dstPath+'/'+name)
        
        
    def _print(self, *args):
        if self._verbose:
            print('Y4M Reader:', ' '.join([str(e) for e in args]))

    def decode(self, data):
        assert isinstance(data, bytes)
        self._data += data
        if self._stream_headers is None:
            self._decode_stream_headers()
            if self._stream_headers is not None:
                self._print('detected stream with headers:', self._stream_headers)
        if self._stream_headers is not None:
            frame = self._decode_frame()
            while frame is not None:
                self._print(frame, 'decoded')
                self._callback(frame)
                frame = self._decode_frame()

    def _frame_size(self):
        assert self._stream_headers['C'].startswith('420'), 'only support I420 fourcc'
        return self._stream_headers['W'] * self._stream_headers['H'] * 3 // 2

    def _decode_frame(self):
        if len(self._data) < self._frame_size():  # no point trying to parse
            return None
        toks = self._data.split(b'\n', 1)
        if len(toks) == 1:  # need more data
            self._print('weird: got plenty of data but no frame header found')
            return None
        headers = toks[0].split(b' ')
        assert headers[0] == b'FRAME', 'expected FRAME (got %r)' % headers[0]
        frame_headers = self._stream_headers.copy()
        for header in headers[1:]:
            header = header.decode('ascii')
            frame_headers[header[0]] = header[1:]
        if len(toks[1]) < self._frame_size():  # need more data
            return None
        yuv = toks[1][0:self._frame_size()]
        self._data = toks[1][self._frame_size():]
        self._count += 1
        return Frame(yuv,(self._stream_headers['W'],self._stream_headers['H']) ,frame_headers, self._count - 1)

    def _decode_stream_headers(self):
        toks = self._data.split(b'\n', 1)
        if len(toks) == 1:  # buffer all header data until eof
            return
        self._stream_headers = {}
        self._data = toks[1]  # save the beginning of the stream for later
        headers = toks[0].split(b' ')
        assert headers[0] == b'YUV4MPEG2', 'unknown type %s' % headers[0]
        for header in headers[1:]:
            header = header.decode('ascii')
            self._stream_headers[header[0]] = header[1:]
        assert 'W' in self._stream_headers, 'No width header'
        assert 'H' in self._stream_headers, 'No height header'
        assert 'F' in self._stream_headers, 'No frame-rate header'
        self._stream_headers['W'] = int(self._stream_headers['W'])
        self._stream_headers['H'] = int(self._stream_headers['H'])
        self._stream_headers['F'] = [int(n) for n in self._stream_headers['F'].split(':')]
        if 'A' in self._stream_headers:
            self._stream_headers['A'] = [int(n) for n in self._stream_headers['A'].split(':')]
        if 'C' not in self._stream_headers:
            self._stream_headers['C'] = '420jpeg'  # man yuv4mpeg
