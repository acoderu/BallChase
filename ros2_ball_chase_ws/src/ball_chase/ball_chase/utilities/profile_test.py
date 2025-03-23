import pstats
from pstats import SortKey

# Load the stats
p = pstats.Stats('depth_profile.prof')

# Remove the long path names
p.strip_dirs()

# Sort the statistics by cumulative time (most expensive functions)
print("\n---- Top 20 Functions by Cumulative Time ----")
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

# Sort by total time spent in the function itself
print("\n---- Top 20 Functions by Total Time ----")
p.sort_stats(SortKey.TIME).print_stats(20)

# Look at specific functions of interest
print("\n---- Callers of get_3d_position ----")
p.print_callers('get_3d_position')

print("\n---- Callers of _get_reliable_depth ----")
p.print_callers('_get_reliable_depth')

print("\n---- Callees of depth_callback ----")
p.print_callees('depth_callback')
