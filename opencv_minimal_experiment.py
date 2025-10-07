# import cv2
# from pygrabber.dshow_graph import FilterGraph

# graph = FilterGraph()
# cams = graph.get_input_devices()

from pygrabber.dshow_graph import FilterGraph
import cv2

# Step 1: List all connected webcams
graph = FilterGraph()
cams = graph.get_input_devices()  # returns a list of device names
print("Available cameras:")
for i, cam in enumerate(cams):
    print(f"{i}: {cam}")

