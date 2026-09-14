"""Weights: the released checkpoints, read without PyTorch.

A `.ckpt` is a PyTorch zip archive: `data.pkl` holds the object tree, and every tensor's
storage is a separate raw little-endian file next to it. The unpickler below resolves only
the handful of globals a state dict needs -- `OrderedDict`, the typed storages and the
tensor rebuild functions -- and refuses everything else, so loading a checkpoint never
executes arbitrary code.

Parameters are returned as a flat dict keyed by the checkpoint's own names
(`decoder.lin3.rf_weight`, ...). Integer and boolean buffers are dropped: they are the
part decomposition, which `assets.py` rebuilds from the model file.
"""
from __future__ import annotations

import os
import pickle
import urllib.request
import zipfile
from collections import OrderedDict

import jax.numpy as jnp
import numpy as np

RELEASED_URL = ('https://github.com/markomih/VolumetricSMPL/blob/dev/models/'
                'VolumetricSMPL_smplx_{gender}.ckpt?raw=true')

CACHE_DIR = os.environ.get('VOLUMETRICSMPL_CACHE',
                           os.path.join(os.path.expanduser('~'), '.cache', 'VolumetricSMPL'))

_STORAGE_DTYPES = {
    'DoubleStorage': np.float64, 'FloatStorage': np.float32, 'HalfStorage': np.float16,
    'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
}


class _Storage:
    def __init__(self, name: str):
        self.dtype = np.dtype(_STORAGE_DTYPES[name])


def _rebuild_tensor(storage, offset, size, stride, *_):
    size, stride = tuple(size), tuple(stride)
    if any(n == 0 for n in size):
        return np.empty(size, dtype=storage.dtype)
    last = offset + sum((n - 1) * s for n, s in zip(size, stride))
    if offset < 0 or last >= storage.shape[0]:
        raise pickle.UnpicklingError('tensor view exceeds its storage')
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=size,
        strides=tuple(s * storage.dtype.itemsize for s in stride))
    return np.array(view)


def _rebuild_parameter(data, *_):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if module == 'torch' and name in _STORAGE_DTYPES:
            return _Storage(name)
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return _rebuild_tensor
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return _rebuild_parameter
        raise pickle.UnpicklingError(f'unsupported object in checkpoint: {module}.{name}')

    def persistent_load(self, pid):
        kind, storage_type, key, _location, _numel = pid
        if kind != 'storage' or not isinstance(storage_type, _Storage):
            raise pickle.UnpicklingError(f'unsupported persistent id: {pid!r}')
        raw = self._archive.read(f'{self._prefix}/data/{key}')
        return np.frombuffer(raw, dtype=storage_type.dtype.newbyteorder('<'))


def read_checkpoint(path: str):
    """The object tree of a PyTorch `.ckpt`/`.pt` zip archive, with tensors as numpy."""
    if not zipfile.is_zipfile(path):
        raise ValueError(f'{path} is not a zip-format PyTorch checkpoint')
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        prefix = pkl[:-len('/data.pkl')]
        order = f'{prefix}/byteorder'
        if order in archive.namelist() and archive.read(order).strip() != b'little':
            raise ValueError(f'{path} is big-endian, which is not supported')
        with archive.open(pkl) as f:
            return _Unpickler(f, archive, prefix).load()


def released_checkpoint(gender: str = 'neutral', cache_dir: str = CACHE_DIR) -> str:
    """Path to the released SMPL-X checkpoint, downloading it on first use."""
    path = os.path.join(cache_dir, f'VolumetricSMPL_smplx_{gender}.ckpt')
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        url = RELEASED_URL.format(gender=gender)
        print(f'downloading {url}')
        urllib.request.urlretrieve(url, path + '.part')
        os.replace(path + '.part', path)
    return path


def load_weights(spec: str = 'released', gender: str = 'neutral') -> dict:
    """Parameter name -> device array.

    `spec` is `'released'`, a PyTorch `.ckpt` (a Lightning checkpoint or a bare state
    dict), or an `.npz` such as a checkpoint written by `training.train`.
    """
    if spec.endswith('.npz'):
        with np.load(spec) as npz:
            return {k: jnp.asarray(npz[k]) for k in npz.files}
    path = released_checkpoint(gender) if spec == 'released' else spec
    state = read_checkpoint(path)
    state = state.get('state_dict', state)
    return {k: jnp.asarray(v) for k, v in state.items()
            if np.issubdtype(v.dtype, np.floating)}


def save_weights(params: dict, path: str) -> None:
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})
