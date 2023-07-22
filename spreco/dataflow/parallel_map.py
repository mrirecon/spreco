# -*- coding: utf-8 -*-
# File: parallel_map.py
import threading

from six.moves import queue

from spreco.dataflow.concurrency import StoppableThread
from spreco.dataflow.base import DataFlow, DataFlowReentrantGuard, ProxyDataFlow
from spreco.dataflow.common import RepeatedData


__all__ = ['MultiThreadMapData']


class _ParallelMapData(ProxyDataFlow):
    def __init__(self, ds, buffer_size, strict=False):
        super(_ParallelMapData, self).__init__(ds)
        assert buffer_size > 0, buffer_size
        self._buffer_size = buffer_size
        self._buffer_occupancy = 0  # actual #elements in buffer, only useful in strict mode
        self._strict = strict

    def reset_state(self):
        super(_ParallelMapData, self).reset_state()
        if not self._strict:
            ds = RepeatedData(self.ds, -1)
        else:
            ds = self.ds
        self._iter = ds.__iter__()

    def _recv(self):
        pass

    def _send(self, dp):
        pass

    def _recv_filter_none(self):
        ret = self._recv()
        assert ret is not None, \
            "[{}] Map function cannot return None when strict mode is used.".format(type(self).__name__)
        return ret

    def _fill_buffer(self, cnt=None):
        if cnt is None:
            cnt = self._buffer_size - self._buffer_occupancy
        try:
            for _ in range(cnt):
                dp = next(self._iter)
                self._send(dp)
        except StopIteration:
            raise RuntimeError(
                "[{}] buffer_size cannot be larger than the size of the DataFlow when strict=True! "
                "Please use a smaller buffer_size!".format(type(self).__name__))
        self._buffer_occupancy += cnt

    def get_data_non_strict(self):
        for dp in self._iter:
            self._send(dp)
            ret = self._recv()
            if ret is not None:
                yield ret

    def get_data_strict(self):
        self._fill_buffer()
        for dp in self._iter:
            self._send(dp)
            yield self._recv_filter_none()
        self._iter = self.ds.__iter__()   # refresh

        # first clear the buffer, then fill
        for k in range(self._buffer_size):
            dp = self._recv_filter_none()
            self._buffer_occupancy -= 1
            if k == self._buffer_size - 1:
                self._fill_buffer()
            yield dp

    def __iter__(self):
        if self._strict:
            yield from self.get_data_strict()
        else:
            yield from self.get_data_non_strict()


class MultiThreadMapData(_ParallelMapData):
    """
    Same as :class:`MapData`, but start threads to run the mapping function.
    This is useful when the mapping function is the bottleneck, but you don't
    want to start processes for the entire dataflow pipeline.

    The semantics of this class is **identical** to :class:`MapData` except for the ordering.
    Threads run in parallel and can take different time to run the
    mapping function. Therefore the order of datapoints won't be preserved.

    When ``strict=True``, ``MultiThreadMapData(df, ...)``
    is guaranteed to produce the exact set of data as ``MapData(df, ...)``,
    if both are iterated until ``StopIteration``. But the produced data will have different ordering.
    The behavior of strict mode is undefined if the given dataflow ``df`` is infinite.

    When ``strict=False``, the data that's produced by ``MultiThreadMapData(df, ...)``
    is a reordering of the data produced by ``RepeatedData(MapData(df, ...), -1)``.
    In other words, first pass of ``MultiThreadMapData.__iter__`` may contain
    datapoints from the second pass of ``df.__iter__``.


    Note:
        1. You should avoid starting many threads in your main process to reduce GIL contention.

           The threads will only start in the process which calls :meth:`reset_state()`.
           Therefore you can use ``MultiProcessRunnerZMQ(MultiThreadMapData(...), 1)``
           to reduce GIL contention.
    """
    class _Worker(StoppableThread):
        def __init__(self, inq, outq, evt, map_func):
            super(MultiThreadMapData._Worker, self).__init__(evt)
            self.inq = inq
            self.outq = outq
            self.func = map_func
            self.daemon = True

        def run(self):
            try:
                while True:
                    dp = self.queue_get_stoppable(self.inq)
                    if self.stopped():
                        return
                    # cannot ignore None here. will lead to unsynced send/recv
                    obj = self.func(dp)
                    self.queue_put_stoppable(self.outq, obj)
            except Exception:
                if self.stopped():
                    pass        # skip duplicated error messages
                else:
                    raise
            finally:
                self.stop()

    def __init__(self, ds, num_thread=None, map_func=None, *, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            num_thread (int): number of threads to use
            map_func (callable): datapoint -> datapoint | None. Return None to
                discard/skip the datapoint.
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        if strict:
            # In strict mode, buffer size cannot be larger than the total number of datapoints
            try:
                buffer_size = min(buffer_size, len(ds))
            except Exception:  # ds may not have a length
                pass

        super(MultiThreadMapData, self).__init__(ds, buffer_size, strict)
        assert num_thread > 0, num_thread

        self._strict = strict
        self.num_thread = num_thread
        self.map_func = map_func
        self._threads = []
        self._evt = None

    def reset_state(self):
        super(MultiThreadMapData, self).reset_state()
        if self._threads:
            self._threads[0].stop()
            for t in self._threads:
                t.join()

        self._in_queue = queue.Queue()
        self._out_queue = queue.Queue()
        self._evt = threading.Event()
        self._threads = [MultiThreadMapData._Worker(
            self._in_queue, self._out_queue, self._evt, self.map_func)
            for _ in range(self.num_thread)]
        for t in self._threads:
            t.start()

        self._guard = DataFlowReentrantGuard()

        # Call once at the beginning, to ensure inq+outq has a total of buffer_size elements
        self._fill_buffer()

    def _recv(self):
        return self._out_queue.get()

    def _send(self, dp):
        self._in_queue.put(dp)

    def __iter__(self):
        with self._guard:
            yield from super(MultiThreadMapData, self).__iter__()

    def __del__(self):
        if self._evt is not None:
            self._evt.set()
        for p in self._threads:
            p.stop()
            p.join(timeout=5.0)
            # if p.is_alive():
            #     logger.warn("Cannot join thread {}.".format(p.name))


if __name__ == '__main__':
    import time

    class Zero(DataFlow):
        def __init__(self, size):
            self._size = size

        def __iter__(self):
            for k in range(self._size):
                yield [k]

        def __len__(self):
            return self._size

    def f(x):
        if x[0] < 10:
            time.sleep(2)
        return x

    ds = Zero(100)
    ds = MultiThreadMapData(ds, 50, f, buffer_size=50, strict=True)
    ds.reset_state()
    for idx, k in enumerate(ds):
        print("Bang!", k, idx)
        if idx == 100:
            break
    print("END!")
