from shapely.geometry import LineString
import matplotlib.pyplot as plt

def are_lines_intersecting(line1_coords, line2_coords):
    line1 = LineString(line1_coords)
    line2 = LineString(line2_coords)
    
    return line1.intersects(line2)

# Define the coordinates of two line segments as (x, y) tuples
line1_coords = [(1, 1), (3, 3)]
line2_coords = [(2, 0), (2, 4)]

# Check if the two lines are intersecting
intersecting = are_lines_intersecting(line1_coords, line2_coords)

# Create a Matplotlib Line object for the first line segment
line1 = plt.Line2D(*zip(*line1_coords), color='green', linewidth=2, marker='o', markersize=8)

# Create a Matplotlib Line object for the second line segment
line2 = plt.Line2D(*zip(*line2_coords), color='blue', linewidth=2, marker='s', markersize=8)

# Create a plot for the lines
fig, ax = plt.subplots()
ax.add_line(line1)
ax.add_line(line2)

# Set axis limits
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)

# Display the result
if intersecting:
    result = "The lines are intersecting."
else:
    result = "The lines are not intersecting."

ax.set_title(result)
plt.show()
