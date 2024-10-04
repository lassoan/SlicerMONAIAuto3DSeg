

def humanReadableTimeFromSec(seconds):
  import math
  if not seconds:
    return "N/A"
  if seconds < 55:
    # if less than a minute, round up to the nearest 5 seconds
    return f"{math.ceil(seconds / 5) * 5} sec"
  elif seconds < 60 * 60:
    # if less than 1 hour, round up to the nearest minute
    return f"{math.ceil(seconds / 60)} min"
  # Otherwise round up to the nearest 0.1 hour
  return f"{seconds / 3600:.1f} h"
