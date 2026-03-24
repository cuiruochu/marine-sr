"""测试回调系统。"""

from src.core.callbacks import Callback, CallbackList


class SimpleTestCallback(Callback):
    def __init__(self):
        self.train_begin_called = False
        self.train_end_called = False
        self.epoch_begin_called = False
        self.epoch_end_called = False

    def on_train_begin(self, engine):
        self.train_begin_called = True

    def on_train_end(self, engine):
        self.train_end_called = True

    def on_epoch_begin(self, engine, epoch):
        self.epoch_begin_called = True

    def on_epoch_end(self, engine, epoch, logs):
        self.epoch_end_called = True


def test_callback_lifecycle():
    callback = SimpleTestCallback()
    cb_list = CallbackList([callback])

    cb_list.on_train_begin(None)
    assert callback.train_begin_called

    cb_list.on_epoch_begin(None, 1)
    assert callback.epoch_begin_called

    cb_list.on_epoch_end(None, 1, {})
    assert callback.epoch_end_called

    cb_list.on_train_end(None)
    assert callback.train_end_called


def test_callback_state_dict():
    callback = SimpleTestCallback()

    state = callback.state_dict()
    assert isinstance(state, dict)

    callback.load_state_dict({"test": 1})
