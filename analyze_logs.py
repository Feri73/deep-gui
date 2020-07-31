import argparse
import glob
import os
import pickle
import shutil
from collections import defaultdict
from functools import partial
from shutil import copyfile
from typing import List, Dict, Callable, Tuple, Union, Optional, Any

import numpy as np
import scipy
from scipy import stats
import tensorflow as tf

Logs = Dict[str, np.ndarray]
DimVals = List[Union[str, List[str]]]
AnalysisResult = Tuple[Tuple[Union[Logs, Tuple[Logs, Logs]], str, DimVals], ...]


def read_logs(logs_dir: str, tags: List[str], tools: List[str], apps: List[str], runs_per_app: int,
              runs_per_app_per_tester: int, error_on_missing: bool = True, from_cache: bool = False,
              excluded_apps: List[str] = None, excluded_app_run_nums: List[int] = None,
              excluded_data_indices: List[int] = None) -> Logs:
    """
    Assumes each chunk only includes data for 1 app and apps are run in a certain order shared among  all tools
    :param apps: before exclusion
    :param excluded_app_run_nums: for each tester of a tool, this works separately
    :param runs_per_app: after exclusion. the sum of runs per an app in testers of a tool
    :return [tag][tool][app][app_run][time]
    """
    if from_cache:
        print('loading from cache')
        with open('.analysis_cache.pck', 'rb') as file:
            return pickle.load(file)

    excluded_apps = excluded_apps or []
    excluded_app_run_nums = excluded_app_run_nums or []
    excluded_data_indices = excluded_data_indices or []
    result = defaultdict(lambda: [[[{} for ___ in range(runs_per_app)] for __ in apps] for _ in tools])
    tools_testers = defaultdict(list)
    for run in os.listdir(logs_dir):
        if 'chunk' not in run:
            continue
        tool = run.split('_tester')[0]
        if tool not in tools:
            continue
        tester = run.split('_chunk')[0]
        if tester not in tools_testers[tool]:
            tools_testers[tool].append(tester)
        tool_index = tools.index(tool)
        chunk = int(run.split('chunk_')[1])
        app_index = chunk % len(apps)
        if apps[app_index] in excluded_apps:
            continue
        app_run_num = chunk // len(apps)
        if app_run_num in excluded_app_run_nums:
            continue
        # this line is for merging same tool records when run in different testers
        app_run_num += runs_per_app_per_tester * tools_testers[tool].index(tester)
        run_path = f'{logs_dir}/{run}'
        for event_file in os.listdir(run_path):
            print('starting', run)
            tmp_file = '.tmp_event'
            copyfile(f'{run_path}/{event_file}', tmp_file)
            for event in tf.train.summary_iterator(tmp_file):
                for value in event.summary.value:
                    if value.tag not in tags:
                        continue
                    result[value.tag][tool_index][app_index][app_run_num][event.step] = value.simple_value
            os.remove(tmp_file)

    #apps = [a for a in apps if a not in excluded_apps]

    for tag in tags:
        for tool_i in range(len(tools)):
            for app_i in range(len(apps)):
                for run_i in range(runs_per_app):
                    plot_dict = result[tag][tool_i][app_i][run_i]
                    result[tag][tool_i][app_i][run_i] = [plot_dict[t]
                                                         for i, t in enumerate(sorted(list(plot_dict.keys())))
                                                         if i not in excluded_data_indices]

    for tag in tags:
        lengths = [len(result[tag][tool_i][app_i][run_i]) for run_i in range(runs_per_app)
                   for app_i in range(len(apps)) for tool_i in range(len(tools))]
        max_length = max(lengths)
        if error_on_missing:
            assert np.all(np.array(lengths) == max_length)
        else:
            for tool_i, tool in enumerate(tools):
                for app_i in range(len(apps)):
                    for run_i in range(runs_per_app):
                        cur_len = len(result[tag][tool_i][app_i][run_i])
                        if cur_len != max_length:
                            print(f'Correcting for tag={tag}, tool={tool}, app_i={app_i}, run_i={run_i},'
                                  f' cur_length={cur_len}, max_length={max_length}')
                            result[tag][tool_i][app_i][run_i][cur_len:max_length] = \
                                [np.nan for _ in range(max_length - cur_len)]

    result = {key: np.array(value) for key, value in result.items()}
    with open('.analysis_cache.pck', 'wb') as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def _write_values(tag: str, values: np.ndarray, errors: Optional[np.ndarray], dim_vals: DimVals,
                  dim_val_is: List[int], writers: Dict[str, tf.summary.FileWriter],
                  logs_dir: str, analysis: str) -> None:
    if values.ndim == 0 or dim_vals[len(dim_val_is)] is None:
        if values.ndim == 0:
            values = [values]
            if errors is not None:
                errors = [errors]
        for i, v in enumerate(values):
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=v)
            if errors is not None:
                summary.value.add(tag=tag, simple_value=v + errors[i])
                summary.value.add(tag=tag, simple_value=v - errors[i])
                summary.value.add(tag=tag, simple_value=v)
            key = analysis + '_' + '_'.join([f'{dim_vals[j]}{d_i}' if isinstance(dim_vals[j], str)
                                             else dim_vals[j][d_i] for j, d_i in enumerate(dim_val_is)])
            if key not in writers:
                path = f'{logs_dir}/{key}'
                if os.path.isdir(path):
                    shutil.rmtree(path)
                writers[key] = tf.summary.FileWriter(path)
            writers[key].add_summary(summary, i)
    else:
        for i, v in enumerate(values):
            _write_values(tag, v, None if errors is None else errors[i],
                          dim_vals, dim_val_is + [i], writers, logs_dir, analysis)


def write_logs(logs: Logs, error_logs: Optional[Logs], dim_vals: List[List[str]], logs_dir: str, analysis: str):
    writers = {}
    for tag, values in logs.items():
        for i, vs in enumerate(dim_vals[:-1]):
            assert isinstance(vs, str) or values.shape[i] == len(vs)
        assert dim_vals[-1] is None or isinstance(dim_vals[-1], str) or values.shape[-1] == len(dim_vals[-1])
        _write_values(tag, values, None if error_logs is None else error_logs[tag],
                      dim_vals, [], writers, logs_dir, analysis)
    for writer in writers.values():
        writer.close()


def move_axes_to_end(a: np.ndarray, axes: List[int], inverse: bool = False) -> np.ndarray:
    if inverse:
        return np.moveaxis(a, [-i - 1 for i in range(len(axes))], axes)
    return np.moveaxis(a, axes, [-i - 1 for i in range(len(axes))])


def process_logs(logs: Logs, func: Callable, axes: List[int],
                 reduce: bool = False, keep_shape: bool = False, **kwargs) -> Logs:
    """
    :param func: Should retain the ndim of the input
    """

    def reshape(data: np.ndarray) -> Tuple[tuple, np.ndarray]:
        swapped_data = move_axes_to_end(data, axes)
        swapped_shape = swapped_data.shape
        reshaped_data = swapped_data.reshape((*swapped_shape[:-len(axes)], -1))
        return swapped_shape, reshaped_data

    result_logs = {}
    for tag in logs.keys():
        swapped_shape, reshaped_log = reshape(logs[tag])
        p_kwargs = {}
        for params_name in kwargs:
            params = kwargs[params_name]
            _, p_reshaped_log = reshape(np.ones_like(logs[tag]) * params[tag])
            p_kwargs[params_name] = p_reshaped_log
        result_log = func(reshaped_log, **p_kwargs)
        if reduce:
            swapped_shape = (*swapped_shape[:-len(axes)],) + (1,) * len(axes)
        result_logs[tag] = result_log.reshape(swapped_shape)
        result_logs[tag] = move_axes_to_end(result_logs[tag], axes, inverse=True)
        if reduce and not keep_shape:
            result_logs[tag] = result_logs[tag].reshape(
                [x for i, x in enumerate(result_logs[tag].shape) if i not in axes])
    return result_logs


def generate_all_coords(axes: List[List[Any]], axis_i: int = 0) -> List[Any]:
    if axis_i >= len(axes):
        yield []
        return
    for v in axes[axis_i]:
        for h in generate_all_coords(axes, axis_i + 1):
            yield [v] + h


def nankstest(array: np.ndarray, axis: int, test_axis: int, test_ref_index: int,
              alternative: str, cutoff: float = .05) -> np.ndarray:
    shape = array.shape
    axis = axis % len(shape)
    test_axis = test_axis % len(shape)
    assert shape[test_axis] == 2
    result = np.zeros(tuple([1 if i == axis else shape[i] for i in range(len(shape))]))
    for coord0 in generate_all_coords([[None] if i == axis or i == test_axis else list(range(shape[i]))
                                       for i in range(len(shape))]):
        coord1 = coord0.copy()
        coord0[test_axis] = 1 - test_ref_index
        coord1[test_axis] = test_ref_index
        coord0 = tuple(coord0)
        coord1 = tuple(coord1)
        array0 = array[coord0]
        array0 = array0[np.logical_not(np.isnan(array0))]
        array1 = array[coord1]
        array1 = array1[np.logical_not(np.isnan(array1))]
        if len(array0) == 0 or len(array1) == 0:
            test_result = 1
        else:
            test_result = scipy.stats.kstest(array0, array1, alternative=alternative)[1]
        result[coord0] = np.nan if test_result < cutoff else 0
        result[coord1] = result[coord0]
    return result


def nanwmean(array: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    weights = np.ones_like(array) * weights
    array = array.copy()
    inds = np.isnan(array)
    array[inds] = 0
    weights[inds] = 0.0000001
    return np.average(array, axis=axis, weights=weights)


def nanwstd(array: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    avg = nanwmean(array, weights, axis)
    return nanwmean((array - np.expand_dims(avg, axis=axis)) ** 2, weights, axis) ** .5


def nanwerror(array: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    weights = np.ones_like(array) * weights
    inds = np.isnan(array)
    weights[inds] = 0.0000001
    return nanwstd(array, weights, axis) / np.sum(weights, axis=-1) ** .5


def zscore_logs(logs: Logs, axes: List[int], **kwargs) -> Logs:
    return process_logs(logs, partial(stats.zscore, axis=-1, nan_policy='omit'), axes, **kwargs)


def max_logs(logs: Logs, axes: List[int], **kwargs) -> Logs:
    return process_logs(logs, partial(np.nanmax, axis=-1), axes=axes, reduce=True, **kwargs)


def min_logs(logs: Logs, axes: List[int], **kwargs) -> Logs:
    return process_logs(logs, partial(np.nanmin, axis=-1), axes=axes, reduce=True, **kwargs)


def range_logs(logs: Logs, axes: List[int], **kwargs) -> Logs:
    return process_logs(logs, lambda a: np.maximum(0.00001, np.nanmax(a, axis=-1) - np.nanmin(a, axis=-1)), axes=axes,
                        reduce=True, **kwargs)


def mean_logs(logs: Logs, axes: List[int], weights: Logs = None, **kwargs) -> Logs:
    if weights is None:
        return process_logs(logs, partial(np.nanmean, axis=-1), axes=axes, reduce=True, **kwargs)
    else:
        return process_logs(logs, partial(nanwmean, axis=-1), axes=axes, reduce=True, weights=weights, **kwargs)


def std_logs(logs: Logs, axes: List[int], weights: Logs = None, **kwargs) -> Logs:
    if weights is None:
        return process_logs(logs, partial(np.nanstd, axis=-1), axes=axes, reduce=True, **kwargs)
    else:
        return process_logs(logs, partial(nanwstd, axis=-1), axes=axes, reduce=True, weights=weights, **kwargs)


def error_logs(logs: Logs, axes: List[int], weights: Logs = None, **kwargs) -> Logs:
    if weights is None:
        return process_logs(logs, lambda a: np.nanstd(a, axis=-1) / np.count_nonzero(~np.isnan(a), axis=-1) ** .5,
                            axes=axes, reduce=True, **kwargs)
    else:
        return process_logs(logs, partial(nanwerror, axis=-1), axes=axes, reduce=True, weights=weights, **kwargs)


def kstest_logs(logs: Logs, axes: List[int], test_axis: int, test_ref_index: int, alternative: str,
                weights: Logs = None, **kwargs) -> Logs:
    if weights is not None:
        raise NotImplementedError('cannot compute weighted ks test')
    return process_logs(logs, partial(nankstest, axis=-1, test_axis=test_axis - sum([a < test_axis for a in axes]),
                                      test_ref_index=test_ref_index, alternative=alternative),
                        axes=axes, reduce=True, **kwargs)


def simple_analysis(logs: Logs, args: argparse.Namespace) -> AnalysisResult:
    parser = argparse.ArgumentParser('Simple Analysis')
    parser.add_argument('--norm-type', action='store', type=str, choices=['mean', 'zscore'])
    parser.add_argument('--norm-ref', action='store', type=str, choices=args.tools)
    parser.add_argument('--norm-axes', action='store', nargs='+', type=str)
    parser.add_argument('--summary-action', action='append', type=str, required=True)
    parser.add_argument('--summary-axes', action='append', nargs='+', type=str, required=True)
    # only one param per action is supported for now
    parser.add_argument('--summary-param', action='append', type=str)
    parser.add_argument('--kstest-alt', action='store', type=str, choices=['less', 'greater', 'two-sided'])
    parser.add_argument('--kstest-ref', action='store', type=str, choices=args.tools)
    args = parser.parse_args(args.args[1:], namespace=args)
    assert (args.norm_type is not None) == (args.norm_axes is not None)
    assert args.norm_type is None or args.norm_ref is None or 'tool' not in args.norm_axes

    dims = ['tool', 'app', 'run', 'time']
    dim_vals = [args.tools, args.apps, 'run', None]

    if args.norm_type is not None:
        norm_axes = sorted([dims.index(axis) for axis in args.norm_axes])

        ref_mean = mean_logs(logs, axes=norm_axes, keep_shape=True)
        ref_std = std_logs(logs, axes=norm_axes, keep_shape=True)
        ref_slice = slice(None)
        if args.norm_ref is not None:
            ref_index = args.tools.index(args.norm_ref)
            ref_slice = slice(ref_index, ref_index + 1)
        for tag in logs.keys():
            logs[tag] -= ref_mean[tag][ref_slice]
            if args.norm_type == 'zscore':
                logs[tag] /= np.maximum(ref_std[tag][ref_slice], .01)

    summary = logs
    for i in range(len(args.summary_action)):
        logs = summary
        action = args.summary_action[i]
        axes = args.summary_axes[i]
        if args.summary_param is not None:
            params = args.summary_param[i]
            p_name, p_action, *p_axes = tuple(params.split('-'))
            p_axes = sorted([dims.index(axis) for axis in p_axes])
            p_val = eval(f'{p_action}_logs')(summary, axes=p_axes, keep_shape=True)
            p_dict = {p_name: p_val}
        else:
            p_dict = {}
        summary_axes = sorted([dims.index(axis) for axis in axes])
        summary_dim_vals = [dim_val for i, dim_val in enumerate(dim_vals) if i not in summary_axes]
        summary = eval(f'{action}_logs')(summary, axes=summary_axes, keep_shape=i < len(args.summary_action) - 1,
                                         **p_dict)

    name_prefix = 'simple'
    results = ((summary, name_prefix + '_means', summary_dim_vals),)
    if action == 'mean':
        errors = error_logs(logs, axes=summary_axes, **p_dict)
        results = (results[0], ((summary, errors), name_prefix + '_errors', summary_dim_vals))
        if len(dim_vals[0]) == 2 and 'weights' not in p_dict:
            ks_ref_index = args.tools.index(args.kstest_ref)
            sig = kstest_logs(logs, axes=summary_axes, test_axis=0, test_ref_index=ks_ref_index,
                              alternative=args.kstest_alt, **p_dict)
            results = results + ((sig, name_prefix + '_sig', summary_dim_vals),)
    return results


parser = argparse.ArgumentParser()
parser.add_argument('--analysis', action='store', type=str, required=True)
parser.add_argument('--name', action='store', type=str)
parser.add_argument('--logs-dir', action='store', type=str, required=True)
parser.add_argument('--tags', action='store', nargs='+', type=str, required=True)
parser.add_argument('--tools', action='store', nargs='+', type=str, required=True)
parser.add_argument('--apps-dir', action='store', type=str, required=True)
parser.add_argument('--runs-per-app', action='store', type=int, required=True)
parser.add_argument('--runs-per-app-per-tester', action='store', type=int, required=True)
parser.add_argument('--excluded-apps-dir', action='store', type=str)
parser.add_argument('--ignore-missing', action='store_true')
parser.add_argument('--use-cache', action='store_true')
parser.add_argument('args', action='store', nargs=argparse.REMAINDER, type=str)
args = parser.parse_args()


def get_apks(dir: str) -> List[str]:
    return [os.path.basename(apk)[:-4] for apk in glob.glob(f'{dir}/*.apk')]


args.apps = get_apks(args.apps_dir)
excluded_apps = None if args.excluded_apps_dir is None else get_apks(args.excluded_apps_dir)

logs = read_logs(args.logs_dir, args.tags, args.tools, args.apps, args.runs_per_app, args.runs_per_app_per_tester,
                 error_on_missing=not args.ignore_missing, from_cache=args.use_cache, excluded_apps=excluded_apps)

results = eval(args.analysis + '_analysis')(logs, args)
for all_logs, name, dim_vals in results:
    if isinstance(all_logs, tuple):
        logs, errors = all_logs
    else:
        logs, errors = all_logs, None
    name = f'{name}_{args.name}' if args.name is not None else name
    write_logs(logs, errors, dim_vals, args.logs_dir, name)

