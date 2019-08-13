__all__ = ['Reader']

from .frame import Frame


class Reader(object):
    def __init__(self, callback, verbose=False):
        self._callback = callback
        self._stream_headers = None
        self._data = bytes()
        self._count = 0
        self._verbose = verbose

    def _print(self, *args):
        if self._verbose:
            print('Y4M Reader:', ' '.join([str(e) for e in args]))

    def decode(self, data, save_prefix):
        assert isinstance(data, bytes)
        self._data += data
        if self._stream_headers is None:  # 第一次进来没有头部，先创建
            self._decode_stream_headers()
            if self._stream_headers is not None:
                self._print('detected stream with headers:', self._stream_headers)
        if self._stream_headers is not None:  # 解析出了头部才进行帧解压
            frame = self._decode_frame()  # 一帧帧开始解压
            while frame is not None:
                self._print(frame, 'decoded')
                self._callback(frame, save_prefix)  # 这个_callback是看你想对这个frame怎么操作都可以
                frame = self._decode_frame()

    def _frame_size(self):
        assert self._stream_headers['C'].startswith('420'), 'only support I420 fourcc'
        return self._stream_headers['W'] * self._stream_headers['H'] * 3 // 2

    def _decode_frame(self):
        if len(self._data) < self._frame_size():  # no point trying to parse  # 已经解析完毕了
            return None
        toks = self._data.split(b'\n', 1)  # 这个toks就是[b'FRAME xxx xxx', b'xxxdata']，如果只有b'FRAME'而没有data就报错
        if len(toks) == 1:  # need more data  # 反正就是没有b'FRAME'或者缺少数据体
            self._print('weird: got plenty of data but no frame header found')
            return None
        headers = toks[0].split(b' ')  # 解析每一个参数
        assert headers[0] == b'FRAME', 'expected FRAME (got %r)' % headers[0]  # 第一个一定要是这个
        frame_headers = self._stream_headers.copy()  # 拿一份解析好的头部来获取某些数据
        for header in headers[1:]:  # 明明已经解析过一次了，为了再来一遍？
            header = header.decode('ascii')
            frame_headers[header[0]] = header[1:]  # 拿第一个字母当元素的键
        if len(toks[1]) < self._frame_size():  # need more data  # 数据体的数量小于frame像素个数
            return None
        yuv = toks[1][0:self._frame_size()]  # 截取yuv数据流
        self._data = toks[1][self._frame_size():]  # 重新更新数据体
        self._count += 1  # 大概是第几帧的意思？

        return Frame(yuv, frame_headers, self._count - 1)  # 这个yuv只是一帧一帧的bit流，还是压缩的

    def _decode_stream_headers(self):
        toks = self._data.split(b'\n', 1)  # 分割一次，即得到头部
        if len(toks) == 1:  # buffer all header data until eof
            return
        self._stream_headers = {}
        self._data = toks[1]  # save the beginning of the stream for later
        headers = toks[0].split(b' ')  # 开始提取明文数据
        assert headers[0] == b'YUV4MPEG2', 'unknown type %s' % headers[0]
        for header in headers[1:]:
            header = header.decode('ascii')  # 去掉b''，变成str
            self._stream_headers[header[0]] = header[1:]  # 把key:value分开保存，第一个字母为key
        assert 'W' in self._stream_headers, 'No width header'  # 这些信息要齐全才能进入下一步
        assert 'H' in self._stream_headers, 'No height header'
        assert 'F' in self._stream_headers, 'No frame-rate header'
        self._stream_headers['W'] = int(self._stream_headers['W'])  # 转换格式存进去
        self._stream_headers['H'] = int(self._stream_headers['H'])
        self._stream_headers['F'] = [int(n) for n in self._stream_headers['F'].split(':')]  # 拆分25:1或者30000:1001之类
        if 'A' in self._stream_headers:  # 好像是可有可无的意思？
            self._stream_headers['A'] = [int(n) for n in self._stream_headers['A'].split(':')]
        if 'C' not in self._stream_headers:  # 即使没有也要强行放个C
            self._stream_headers['C'] = '420jpeg'  # man yuv4mpeg
        # 本次提交的类似b'YUV4MPEG2 W1920 H1080 F25:1 Ip A0:0 C420jpeg XYSCSS=420JPEG\n'
        # 训练的类似b'YUV4MPEG2 W480 H270 F24:1 Ip A0:0 C420mpeg2 XYSCSS=420MPEG2\n'
