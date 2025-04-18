import os
import argparse
from typing import List, Dict
from gtest.inline_profiler import *

current_directory = os.getcwd()

parser = argparse.ArgumentParser()

parser.add_argument(
    '-p', '--path',
    type=str,
    default=f'{current_directory}',
    help='output directory of the dumped file',
    required=False
)

parser.add_argument(
    '-d', '--device_id',
    type=str,
    default=f'*',
    help='device indices to be dumped, should be seperated by , e.g. 0,1,2',
    required=False
)

if __name__ == "__main__":
    args = parser.parse_args()

    gw_cxt = GWContext(lazy_init_device = False)
    map_gw_device : Dict[int, GWDevice] = gw_cxt.get_devices()

    if args.device_id != '*':
        list_device_id = args.device_id.split(',')
        tmp_map_gw_device : Dict[int, GWDevice] = {}
        for device_id, gw_device in map_gw_device.items():
            if str(device_id) in list_device_id:
                tmp_map_gw_device[device_id] = gw_device
        map_gw_device = tmp_map_gw_device

    os.chdir(args.path)
    for gw_device in map_gw_device.values():
        gw_device.export_metric_properties(args.path)
