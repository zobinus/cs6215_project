import os
import argparse

from .scheduler import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '-b', '--backend',
    type=str,
    default='cuda',
    help='backend to use',
    choices=['cuda', 'rocm'],
    required=False
)
parser.add_argument(
    '-s', '--script',
    type=str,
    default=f'{os.getcwd()}/WatchScript.py',
    help='WatchScript provided by the user',
    required=False
)
parser.add_argument(
    '-w', '--world-size',
    type=str,
    default=1,
    help='world size (#processes) of the program to be profiled',
    required=False
)
parser.add_argument(
    '-vi', '--visual',
    action='store_true',
    help='Start visualization frontend to exhibit the data',
    required=False
)
parser.add_argument(
    'command',
    type=str,
    nargs=argparse.REMAINDER,
    help='command to be watched'
)

args = parser.parse_args()

scheduler : GWScheduler = GWScheduler(
    backend = args.backend,
    watchscript_path = args.script,
    visual = args.visual,
    world_size = args.world_size,
    command = args.command
)

scheduler.serve()
scheduler.start_capsule()

while True:
    pass
