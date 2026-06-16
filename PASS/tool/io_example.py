import numpy as np
import pandas as pd
import tfs
import h5py


def tfs_read(file_path: str):
    table = tfs.read(file_path)  # TFSDataFrame, which is DataFrame + headers

    headers = table.headers  # get header information such as particle, energy, etc.
    column_names = table.columns  # get column names such as turn, x, etc.
    shape = table.shape  # get data shape (rows, columns)

    turn = table["turn"]  # get specific column
    turn_np = table["turn"].to_numpy()  # get specific column and convert to ndarray


def tfs_write(file_path: str):
    df = pd.DataFrame({
        "turn": [0, 1, 2],
        "x": [1e-4, 2e-4, 3e-4],
        "px": [4e-4, 5e-4, 6e-4],
    })
    headers = {}
    headers["name"] = "test"

    table = tfs.TfsDataFrame(df, headers=headers)
    tfs.write(file_path, table)


def hdf5_read(file_path: str, dataset_name: str):

    with h5py.File(file_path, 'r') as f:
        dset = f[dataset_name]
        data = dset[()]
        columns = list(dset.attrs['columns'])
        idx_type = dset.attrs.get('index_type', 'default')
        if idx_type == 'default':
            index = pd.RangeIndex(len(data))
        else:
            index_vals = dset.attrs['index_values']
            index = pd.Index(index_vals)
        idx_name = dset.attrs.get('index_name')
        if idx_name:
            index.name = idx_name
    return pd.DataFrame(data, columns=columns, index=index)


def hdf5_write(file_path: str, dataset_name: str, df: pd.DataFrame):

    with h5py.File(file_path, 'a') as f:
        dset = f.create_dataset(dataset_name, data=df.values)
        dset.attrs['columns'] = list(df.columns)
        if isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1:
            dset.attrs['index_type'] = 'default'
        else:
            dset.attrs['index_type'] = 'custom'
            idx_vals = df.index.tolist()
            if all(isinstance(v, (int, float)) for v in idx_vals):
                dset.attrs['index_values'] = idx_vals
            else:
                dset.attrs['index_values'] = [str(v) for v in idx_vals]
        if df.index.name:
            dset.attrs['index_name'] = df.index.name
