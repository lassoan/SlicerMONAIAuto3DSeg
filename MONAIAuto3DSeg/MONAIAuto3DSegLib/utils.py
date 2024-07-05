

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


def assignInputNodesByName(inputs, loadedSampleNodes):
  inputNodes = []
  for inputIndex, input in enumerate(inputs):
    namePattern = input.get("namePattern")
    if namePattern:
      matchingNode = findFirstNodeByNamePattern(namePattern, loadedSampleNodes)
    else:
      matchingNode = loadedSampleNodes[inputIndex] if inputIndex < len(loadedSampleNodes) else \
        loadedSampleNodes[0]
    inputNodes.append(matchingNode)
  return inputNodes


def findFirstNodeByNamePattern(namePattern, nodes):
  import fnmatch
  for node in nodes:
    if fnmatch.fnmatchcase(node.GetName(), namePattern):
      return node
  return None