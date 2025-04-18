# file:         watch_occupancy.py
# author:       Zhuobin Huang (zhuobin@u.nus.edu)
# description:  profile and trace low occupancy gpu kernel

import numpy as np
import gtest.script.device as device
import gtest.script.kernel as kernel
import gtest.script.profiler as profiler
import gtest.script.tracer as tracer
import gtest.script.report as report

# report non-ideal launch scale
if kernel.grid_size % (device.num_SMs * kernel.max_block_per_SM) != 0:
    report.WARN("launch scale could cause low occupancy")

# we only profile those kernels that have large static SMEM or large register usage
if  kernel.static_mem_size >= device.SMEM_size_per_SM/4 or kernel.num_regs >= device.num_regs_per_SM:
    # obtain occupancy metric
    occupancy = profiler.watch(
        metrics=['sm__warps_active.avg.pct_of_peak_sustained_active']
    )[0]

    # if occupancy is low, we trace the last 5 kernels
    if occupancy < 0.4:
        blksche_trace = tracer.watch_block_schedule()
        end_timestamps = np.array([blk.end_ts for blk in blksche_trace])
        lastest_indices = np.argsort(end_timestamps)[-5:][::-1]
        lastest_blks = [blksche_trace[i] for i in lastest_indices]
        report.FAILED(lastest_blks)
    else:
        report.PASS()
