import numpy
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class DatasetStat:
    n: int
    x_min: float
    x_max: float
    x_mean: float
    x_std: float


def compute_stat(ds: Dataset) -> DatasetStat:
    # samples in ds are required to have the same weight

    n = len(ds)

    x_mins = numpy.empty((n,), dtype=numpy.float32)
    x_maxs = numpy.empty((n,), dtype=numpy.float32)
    x_means = numpy.empty((n,), dtype=numpy.float64)
    x_vars = numpy.empty((n,), dtype=numpy.float64)

    def get_stat_idx(i: int):
        x, y, _ = ds[i]
        x_mins[i] = numpy.nanpercentile(x, 0.5)
        x_maxs[i] = numpy.nanpercentile(x, 99.5)
        x = numpy.clip(x, x_mins[i], x_maxs[i])
        x_means[i] = numpy.nanmean(x)
        x_vars[i] = numpy.nanvar(x)

    # for i in range(n):
    #     get_stat_idx(i)

    futs = []
    with ThreadPoolExecutor(max_workers=min(n, 256)) as executor:
        for i in range(n):
            futs.append(executor.submit(get_stat_idx, i))

    assert all(fut.exception() is None for fut in futs)

    x_mean = numpy.mean(x_means)
    x_var = numpy.mean((x_vars + (x_means - x_mean) ** 2))

    return DatasetStat(
        n=n,
        x_min=float(x_mins.mean()),
        x_max=float(x_maxs.mean()),
        x_mean=float(x_mean),
        x_std=float(numpy.sqrt(x_var)),
    )
