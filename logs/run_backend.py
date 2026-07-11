import os
import runpy
import sys

os.chdir(r"C:\MAAV\backend")
sys.path.insert(0, r"C:\MAAV\backend\scripts")
runpy.run_path("scripts/run_dev_server.py", run_name="__main__")
