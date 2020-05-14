import argparse
import os
import pickle
import shutil
from collections import defaultdict
from functools import partial
from shutil import copyfile
from typing import List, Dict, Callable, Tuple, Union, Optional

import numpy as np
from scipy import stats
import tensorflow as tf

Logs = Dict[str, np.ndarray]
AnalysisResult = Tuple[Tuple[Union[Logs, Tuple[Logs, Logs]], str, List[List[str]]], ...]


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

    apps = [a for a in apps if a not in excluded_apps]

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


def _write_values(tag: str, values: np.ndarray, errors: Optional[np.ndarray], dim_vals: List[List[str]],
                  dim_val_is: List[int], writers: Dict[str, tf.summary.FileWriter],
                  logs_dir: str, analysis: str) -> None:
    if values.ndim == 1:
        for i, v in enumerate(values):
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=v)
            if errors is not None:
                summary.value.add(tag=tag, simple_value=v + errors[i])
                summary.value.add(tag=tag, simple_value=v - errors[i])
                summary.value.add(tag=tag, simple_value=v)
            key = analysis + '_' + '_'.join([dim_vals[i][d_i] for i, d_i in enumerate(dim_val_is)])
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
        assert values.shape[:-1] == tuple(len(x) for x in dim_vals)
        _write_values(tag, values, None if error_logs is None else error_logs[tag],
                      dim_vals, [], writers, logs_dir, analysis)
    for writer in writers.values():
        writer.close()


def move_axes_to_end(a: np.ndarray, axes=List[int], inverse: bool = False) -> np.ndarray:
    if inverse:
        return np.moveaxis(a, [-i - 1 for i in range(len(axes))], axes)
    return np.moveaxis(a, axes, [-i - 1 for i in range(len(axes))])


def process_logs(logs: Logs, func: Callable[[np.ndarray], np.ndarray], axes: List[int],
                 reduce: bool = False, keep_shape: bool = False) -> Logs:
    """
    :param func: Should retain the ndim of the input
    """
    result_logs = {}
    for tag in logs.keys():
        swapped_log = move_axes_to_end(logs[tag], axes)
        swapped_shape = swapped_log.shape
        reshaped_log = swapped_log.reshape((*swapped_shape[:-len(axes)], -1))
        result_log = func(reshaped_log)
        if reduce:
            swapped_shape = (*swapped_shape[:-len(axes)],) + (1,) * len(axes)
        result_logs[tag] = result_log.reshape(swapped_shape)
        result_logs[tag] = move_axes_to_end(result_logs[tag], axes, inverse=True)
        if reduce and not keep_shape:
            result_logs[tag] = result_logs[tag].reshape(
                [x for i, x in enumerate(result_logs[tag].shape) if i not in axes])
    return result_logs


def zscore_logs(logs: Logs, axes=List[int], **kwargs) -> Logs:
    return process_logs(logs, partial(stats.zscore, axis=-1, nan_policy='omit'), axes, **kwargs)


def mean_logs(logs: Logs, axes=List[int], **kwargs) -> Logs:
    return process_logs(logs, partial(np.nanmean, axis=-1), axes=axes, reduce=True, **kwargs)


def std_logs(logs: Logs, axes=List[int], **kwargs) -> Logs:
    return process_logs(logs, partial(np.nanstd, axis=-1), axes=axes, reduce=True, **kwargs)


def error_logs(logs: Logs, axes=List[int], **kwargs) -> Logs:
    return process_logs(logs, lambda a: np.nanstd(a, axis=-1) / np.count_nonzero(~np.isnan(a), axis=-1) ** .5,
                        axes=axes, reduce=True, **kwargs)


def simple_analysis(logs: Logs, args: argparse.Namespace) -> AnalysisResult:
    parser = argparse.ArgumentParser('Simple Analysis')
    parser.add_argument('--zscore', action='store_true')
    parser.add_argument('--zscore-ref', action='store', type=str, choices=['all', *args.tools])
    parser.add_argument('--zscore-axes', action='store', nargs='+', type=str)
    args = parser.parse_args(args.args[1:], namespace=args)
    assert args.zscore == (args.zscore_ref is not None) == (args.zscore_axes is not None)
    assert args.zscore_ref == 'all' or 'tool' not in args.zscore_axes

    if args.zscore:
        dims = ['tool', 'app', 'run', 'time']
        axes = [dims.index(axis) for axis in args.zscore_axes]

        if args.zscore_ref == 'all':
            logs = zscore_logs(logs, axes=axes)
        else:
            ref_index = args.tools.index(args.zscore_ref)
            ref_mean = mean_logs(logs, axes=axes, keep_shape=True)
            ref_std = std_logs(logs, axes=axes, keep_shape=True)
            for tag in logs.keys():
                logs[tag] = (logs[tag] - ref_mean[tag][ref_index: ref_index + 1]) / \
                            (ref_std[tag][ref_index: ref_index + 1] + np.finfo(float).eps)

    means = mean_logs(logs, axes=[1, 2])
    errors = error_logs(logs, axes=[1, 2])
    name_prefix = 'simple'
    return ((means, name_prefix + '_means', [args.tools]),
            ((means, errors), name_prefix + '_errors', [args.tools]))


parser = argparse.ArgumentParser()
parser.add_argument('--analysis', action='store', type=str, required=True)
parser.add_argument('--name', action='store', type=str)
parser.add_argument('--logs-dir', action='store', type=str, required=True)
parser.add_argument('--tags', action='store', nargs='+', type=str, required=True)
parser.add_argument('--tools', action='store', nargs='+', type=str, required=True)
parser.add_argument('--apps', action='store', nargs='+', type=str, required=True)
parser.add_argument('--runs-per-app', action='store', type=int, required=True)
parser.add_argument('--runs-per-app-per-tester', action='store', type=int, required=True)
parser.add_argument('--ignore-missing', action='store_true')
parser.add_argument('--use-cache', action='store_true')
parser.add_argument('args', action='store', nargs=argparse.REMAINDER, type=str)
args = parser.parse_args()

logs = read_logs(args.logs_dir, args.tags, args.tools, args.apps, args.runs_per_app, args.runs_per_app_per_tester,
                 error_on_missing=not args.ignore_missing, from_cache=args.use_cache, excluded_data_indices=[0])

results = eval(args.analysis + '_analysis')(logs, args)
for all_logs, name, dim_vals in results:
    if isinstance(all_logs, tuple):
        logs, errors = all_logs
    else:
        logs, errors = all_logs, None
    name = f'{name}_{args.name}' if args.name is not None else name
    write_logs(logs, errors, dim_vals, args.logs_dir, name)
