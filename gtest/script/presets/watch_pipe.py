# file:         watch_pipe.py
# author:       Zhuobin Huang (zhuobin@u.nus.edu)
# description:  profile and trace low occupancy gpu kernel

import numpy as np
from typing import List, Dict
import gtest.context as context
import gtest.tracer as tracer
import gtest.inline_profiler as inline_profiler
import gtest.inline_profiler.METRIC as METRIC
import gtest.tracer.TRACE as TRACE


# offload watch rules
watch_result : Dict[str, inline_profiler.result] = inline_profiler.watch(
    metric = [
        {'fma_avg_trpt', METRIC.pipe.fma.throughput.avg},
        {'tc_avg_trpt', METRIC.pipe.tc.throughput.avg}
    ]
)


# trace suspect kernels that
if watch_result['fma_avg_trpt'] < 0.5:
    # gWatch Contribution 1: lightweightly zoom into which kernel cause the problem
    #                       by Zoom Search (缩圈式搜寻)
    # this phrase would conduct auto-reply to narrow the scope that cause the problem
    kernels : List[context.kernel] = watch_result['fma_avg_trpt'].get_bad_kernels(
        threshold = 0.5, top_k = 5, comparator = '<='
    )

    # no suspect kernel found, this problem could be cpu bound
    if len(kernels) == 0:
        context.warn("no suspect kernel found, this problem could be cpu bound")

    # conduct trace for each suspect kernel
    for k in kernels:
        # gWatch Contribution 2: lightweightly collect trace result
        #                       by program analysis
        trace_result : Dict[str, tracer.result] = tracer.trace(
            kernel = k,
            target = [
                {'block_schedule', TRACE.block_schedule},
                {'warp_schedule', TRACE.warp_schedule},
            ]
        )

        # analyse on block schedule trace result
        for block in trace_result['block_schedule']:
            pass

        # analyse on warp schedule trace result
        for warp in trace_result['warp_schedule']:
            pass
