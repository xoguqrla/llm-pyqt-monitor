# core/workers.py
from __future__ import annotations
from PyQt5.QtCore import QObject, pyqtSignal, QThread

class Worker(QObject):
    finished = pyqtSignal(object, object)  # (result, error)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.finished.emit(res, None)
        except Exception as e:
            self.finished.emit(None, e)

def run_in_thread(parent, fn, callback, *args, **kwargs):
    """Run blocking fn(*args, **kwargs) in a QThread, then callback(result, error)."""
    thread = QThread(parent)
    worker = Worker(fn, *args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    def _finish(res, err):
        try:
            callback(res, err)
        finally:
            thread.quit()
            worker.deleteLater()
            thread.wait()
            thread.deleteLater()
    worker.finished.connect(_finish)
    thread.start()
    return thread  # keep a reference if you need to cancel/track
