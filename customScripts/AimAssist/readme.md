# Performance optimizations

This version aimes to achieve the best performance possible on AMD hardware.
To achieve this, the script acts more as an aim assist instead of a full fledged aimbot.
The user will still need to do most on the aim

Changes that have been made:
- general clean up of the codebase, added some comments, removed duplicate imports
- added some variables to quickly adjust the behavior of the script: offsets for mouse movement, headshot offset, Max_FPS, model selection etc...
- removed garbage collection and  (I have 32gb of ram and I have never run into memory issues)
- added 'so.enable_cpu_mem_arena = True' this should improve latencies
- added 'so.intra_op_num_threads', the lower the threads the less cpu is used per core

## More info
Contact Parideboy