#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#


import os, threading


class PipeCapture:

    ESC = b'\x1b'  # 27

    def __init__(self, stream, isthread=True):
        self._stream = stream
        self._isthread = isthread
        self._descriptor = self._stream.fileno()
        self._pipe_out, self._pipe_in = os.pipe()
        self._worker = None
        self._descriptor_dub = None
        self._publish = None
        self._buffer = []

    def open(self, publish):
        self._publish = publish
        self._descriptor_dub = os.dup(self._descriptor)
        os.dup2(self._pipe_in, self._descriptor)
        if self._isthread:
            self._worker = threading.Thread(target=self.read)
            self._worker.start()

    def close(self):
        if self._publish is None: return
        self._publish = None
        self._stream.write(PipeCapture.ESC.decode('utf-8'))
        self._stream.flush()
        if self._isthread:
            self._worker.join()
        os.close(self._pipe_out)
        os.dup2(self._descriptor_dub, self._descriptor)

    def read(self):
        while self._publish is not None:
            char = os.read(self._pipe_out, 1)
            if char == PipeCapture.ESC: break
            self._buffer.append(char.decode('utf-8'))
            if self._buffer[-1] == '\n':            
                self._publish(''.join(self._buffer))
                self._buffer.clear()
