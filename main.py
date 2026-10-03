import cv2
import csv
import numpy as np
import pydicom
from sklearn.cluster import KMeans

ds = pydicom.dcmread("test.dcm")
img = ds.pixel_array

print("DICOM loaded")
print("Image size:", ds.Rows, "x", ds.Columns)
print("Bits Stored:", ds.BitsStored)
print("Pixel dtype:", img.dtype)
print("Pixel min:", img.min())
print("Pixel max:", img.max())

roi_size = 100

height, width = img.shape

# 12-bit DICOM is kept in img for analysis.
# Use an 8-bit normalized copy for OpenCV display/overlay.
display_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

features = []
positions = []
roi_numbers = []

roi_number = 0

for y in range(0, height - roi_size + 1, roi_size):

    for x in range(0, width - roi_size + 1, roi_size):

        roi = img[
            y:y + roi_size,
            x:x + roi_size
        ]

        roi_number += 1

        mean_value = roi.mean()
        std_value = roi.std()
        min_value = roi.min()
        max_value = roi.max()

        horizontal_diff = np.abs(
            roi[:, 1:].astype(np.int16)
            - roi[:, :-1].astype(np.int16)
        )

        vertical_diff = np.abs(
            roi[1:, :].astype(np.int16)
            - roi[:-1, :].astype(np.int16)
        )

        horizontal_average = horizontal_diff.mean()
        vertical_average = vertical_diff.mean()

        overall_average = (
            horizontal_average + vertical_average
        ) / 2

        features.append([
            mean_value,
            std_value,
            min_value,
            max_value,
            horizontal_average,
            vertical_average,
            overall_average
        ])

        positions.append((x, y))
        roi_numbers.append(roi_number)

data = np.array(features)

mean_values = data.mean(axis=0)
std_values = data.std(axis=0)

standardized = (
    data - mean_values
) / std_values

model = KMeans(
    n_clusters=2,
    random_state=0,
    n_init=10
)

model.fit(standardized)

centers = model.cluster_centers_

boundary_results = []

for i in range(len(standardized)):

    difference1 = standardized[i] - centers[0]
    difference2 = standardized[i] - centers[1]

    distance1 = np.sqrt(
        np.sum(difference1 ** 2)
    )

    distance2 = np.sqrt(
        np.sum(difference2 ** 2)
    )

    distance_difference = abs(
        distance1 - distance2
    )

    boundary_results.append([
        roi_numbers[i],
        distance1,
        distance2,
        distance_difference
    ])

boundary_results.sort(
    key=lambda row: row[3]
)

top_candidates = boundary_results[:6]

candidate_numbers = []

for row in top_candidates:
    candidate_numbers.append(row[0])

print("")
print("Boundary candidates")
print("")

for row in top_candidates:

    print(
        "ROI",
        row[0],
        "G1:",
        round(row[1], 3),
        "G2:",
        round(row[2], 3),
        "Difference:",
        round(row[3], 3)
    )

display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

for i in range(len(roi_numbers)):

    if roi_numbers[i] in candidate_numbers:

        x, y = positions[i]

        cv2.rectangle(
            display,
            (x, y),
            (x + roi_size, y + roi_size),
            (0, 0, 255),
            4
        )

        text = "ROI " + str(roi_numbers[i])

        cv2.putText(
            display,
            text,
            (x + 5, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

# Resize entire image for display

display_width = 900

scale = display_width / width

display_height = int(
    height * scale
)

display = cv2.resize(
    display,
    (display_width, display_height)
)

cv2.imshow(
    "Full Image - Boundary Candidates",
    display
)

print("")
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

cluster_display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

for i in range(len(roi_numbers)):

    x, y = positions[i]

    label = model.labels_[i]

    if label == 0:
        # Group 1
        border_color = (255, 0, 0)   # blue
        text = "G1"

    else:
        # Group 2
        border_color = (0, 255, 0)   # green
        text = "G2"

    cv2.rectangle(
        cluster_display,
        (x, y),
        (x + roi_size, y + roi_size),
        border_color,
        3
    )

    cv2.putText(
        cluster_display,
        text,
        (x + 5, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        border_color,
        2
    )


cluster_display_width = 900

scale = cluster_display_width / width

cluster_display_height = int(
    height * scale
)

cluster_display = cv2.resize(
    cluster_display,
    (
        cluster_display_width,
        cluster_display_height
    )
)

cv2.imshow(
    "Full Image - Group Classification",
    cluster_display
)

print("")
print("Group classification")
print("Blue = Group 1")
print("Green = Group 2")
print("")
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

print("")
print("Group feature comparison")
print("")

feature_names = [
    "Mean",
    "Std",
    "Min",
    "Max",
    "Horizontal",
    "Vertical",
    "Overall"
]

for group in range(2):

    group_data = data[
        model.labels_ == group
    ]

    group_mean = group_data.mean(axis=0)

    print(
        "Group",
        group + 1
    )

    for i in range(len(feature_names)):

        print(
            feature_names[i],
            ":",
            round(group_mean[i], 3)
        )

    print("")

print("")
print("Boundary candidate ROI images")
print("")

for row in top_candidates:

    roi_number = row[0]

    index = roi_numbers.index(roi_number)

    x, y = positions[index]

    roi = img[
        y:y + roi_size,
        x:x + roi_size
    ]

    roi_display = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    enlarged = cv2.resize(
        roi_display,
        (600, 600),
        interpolation=cv2.INTER_NEAREST
    )

    window_name = (
        "Boundary Candidate ROI "
        + str(roi_number)
    )

    cv2.imshow(
        window_name,
        enlarged
    )

print("Press any key on the image windows.")

cv2.waitKey(0)

cv2.destroyAllWindows()

boundary_map = np.zeros(
    (height, width),
    dtype=np.uint8
)

for i in range(len(roi_numbers)):

    x, y = positions[i]

    label = model.labels_[i]

    boundary_map[
        y:y + roi_size,
        x:x + roi_size
    ] = label + 1


# Create display image

boundary_display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

# Draw boundary between different groups

for y in range(0, height - roi_size, roi_size):

    for x in range(0, width - roi_size, roi_size):

        current = boundary_map[y, x]

        right = boundary_map[y, x + roi_size]

        if current != right:

            cv2.line(
                boundary_display,
                (x + roi_size, y),
                (x + roi_size, y + roi_size),
                (0, 0, 255),
                5
            )

        bottom = boundary_map[y + roi_size, x]

        if current != bottom:

            cv2.line(
                boundary_display,
                (x, y + roi_size),
                (x + roi_size, y + roi_size),
                (0, 0, 255),
                5
            )


# Resize for display

boundary_display_width = 900

scale = boundary_display_width / width

boundary_display_height = int(
    height * scale
)

boundary_display = cv2.resize(
    boundary_display,
    (
        boundary_display_width,
        boundary_display_height
    )
)

cv2.imshow(
    "Group Boundary Map",
    boundary_display
)

print("")
print("Red lines = Group boundary")
print("")
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

print("")
print("Boundary candidate coordinates")
print("")

for row in top_candidates:

    roi_number = row[0]

    index = roi_numbers.index(roi_number)

    x, y = positions[index]

    print(
        "ROI",
        roi_number,
        "X:",
        x,
        "Y:",
        y,
        "Difference:",
        round(row[3], 3)
    )

candidate_display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

top_n = 30

candidate_points = boundary_results[:top_n]

for row in candidate_points:

    roi_number = row[0]

    index = roi_numbers.index(roi_number)

    x, y = positions[index]

    center_x = x + roi_size // 2
    center_y = y + roi_size // 2

    cv2.circle(
        candidate_display,
        (center_x, center_y),
        8,
        (0, 0, 255),
        -1
    )

# Resize for display

candidate_display_width = 900

scale = candidate_display_width / width

candidate_display_height = int(
    height * scale
)

candidate_display = cv2.resize(
    candidate_display,
    (
        candidate_display_width,
        candidate_display_height
    )
)

cv2.imshow(
    "Boundary Candidate Points",
    candidate_display
)

print("")
print("Red dots = Top 30 boundary candidates")
print("")
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

grid_rows = height // roi_size
grid_cols = width // roi_size

labels_grid = np.zeros(
    (grid_rows, grid_cols),
    dtype=np.int32
)

for i in range(len(roi_numbers)):

    roi_number = roi_numbers[i]

    index = roi_number - 1

    row = index // grid_cols
    col = index % grid_cols

    labels_grid[row, col] = model.labels_[i]


boundary_display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

boundary_count = 0

for row in range(grid_rows):

    for col in range(grid_cols):

        current = labels_grid[row, col]

        # Check right neighbor

        if col < grid_cols - 1:

            right = labels_grid[row, col + 1]

            if current != right:

                x = (col + 1) * roi_size
                y1 = row * roi_size
                y2 = (row + 1) * roi_size

                cv2.line(
                    boundary_display,
                    (x, y1),
                    (x, y2),
                    (0, 0, 255),
                    3
                )

                boundary_count += 1

        # Check bottom neighbor

        if row < grid_rows - 1:

            bottom = labels_grid[row + 1, col]

            if current != bottom:

                y = (row + 1) * roi_size
                x1 = col * roi_size
                x2 = (col + 1) * roi_size

                cv2.line(
                    boundary_display,
                    (x1, y),
                    (x2, y),
                    (0, 0, 255),
                    3
                )

                boundary_count += 1

print("")
print("Adjacent group boundaries:", boundary_count)
print("")

# Resize

display_width = 900

scale = display_width / width

display_height = int(
    height * scale
)

boundary_display = cv2.resize(
    boundary_display,
    (
        display_width,
        display_height
    )
)

cv2.imshow(
    "Adjacent Group Boundaries",
    boundary_display
)

print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

boundary_strengths = []

threshold = 15

rows = height // roi_size
cols = width // roi_size

for row in range(rows):

    for col in range(cols):

        current = labels_grid[row, col]

        if col < cols - 1:

            right = labels_grid[row, col + 1]

            if current != right:

                x = (col + 1) * roi_size

                left_area = img[
                    row * roi_size:(row + 1) * roi_size,
                    x - 5:x
                ]

                right_area = img[
                    row * roi_size:(row + 1) * roi_size,
                    x:x + 5
                ]

                left_mean = left_area.mean()
                right_mean = right_area.mean()

                intensity_difference = abs(
                    left_mean - right_mean
                )

                boundary_strengths.append([
                    "Vertical",
                    x,
                    row * roi_size,
                    intensity_difference
                ])

        if row < rows - 1:

            bottom = labels_grid[row + 1, col]

            if current != bottom:

                y = (row + 1) * roi_size

                top_area = img[
                    y - 5:y,
                    col * roi_size:(col + 1) * roi_size
                ]

                bottom_area = img[
                    y:y + 5,
                    col * roi_size:(col + 1) * roi_size
                ]

                top_mean = top_area.mean()
                bottom_mean = bottom_area.mean()

                intensity_difference = abs(
                    top_mean - bottom_mean
                )

                boundary_strengths.append([
                    "Horizontal",
                    col * roi_size,
                    y,
                    intensity_difference
                ])

boundary_strengths.sort(
    key=lambda row: row[3],
    reverse=True
)

print("")
print("Top Boundary Strengths")
print("")

for row in boundary_strengths[:20]:

    print(
        row[0],
        "X:",
        row[1],
        "Y:",
        row[2],
        "Difference:",
        round(row[3], 3)
    )

print("")
print(
    "Total candidate boundaries:",
    len(boundary_strengths)
)

print("")
print("Boundary Strength Distribution")
print("")

ranges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 10),
    (10, 20)
]

for lower, upper in ranges:

    count = 0

    for row in boundary_strengths:

        difference = row[3]

        if lower <= difference < upper:
            count += 1

    print(
        lower,
        "-",
        upper,
        ":",
        count
    )

boundary_display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

threshold = 4

for row in boundary_strengths:

    difference = row[3]

    if difference >= threshold:

        direction = row[0]
        x = row[1]
        y = row[2]

        if direction == "Vertical":

            cv2.line(
                boundary_display,
                (x, y),
                (x, y + roi_size),
                (0, 0, 255),
                4
            )

        else:

            cv2.line(
                boundary_display,
                (x, y),
                (x + roi_size, y),
                (0, 0, 255),
                4
            )

display_width = 900

scale = display_width / width

display_height = int(
    height * scale
)

boundary_display = cv2.resize(
    boundary_display,
    (
        display_width,
        display_height
    )
)

cv2.imshow(
    "Strong Boundaries",
    boundary_display
)

print("")
print("Threshold:", threshold)
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

# Label strong boundaries

label_display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

threshold = 4

label_number = 1

for row in boundary_strengths:

    difference = row[3]

    if difference >= threshold:

        direction = row[0]
        x = row[1]
        y = row[2]

        if direction == "Vertical":

            cv2.line(
                label_display,
                (x, y),
                (x, y + roi_size),
                (0, 0, 255),
                4
            )

            cv2.putText(
                label_display,
                str(label_number),
                (x + 8, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            cv2.line(
                label_display,
                (x, y),
                (x + roi_size, y),
                (0, 0, 255),
                4
            )

            cv2.putText(
                label_display,
                str(label_number),
                (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        label_number += 1

display_width = 900

scale = display_width / width

display_height = int(
    height * scale
)

label_display = cv2.resize(
    label_display,
    (
        display_width,
        display_height
    )
)

cv2.imshow(
    "Numbered Strong Boundaries",
    label_display
)

print("")
print("Strong boundaries:", label_number - 1)
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

# Analyze neighboring strong boundaries

threshold = 4

strong_boundaries = []

for row in boundary_strengths:

    difference = row[3]

    if difference >= threshold:

        direction = row[0]
        x = row[1]
        y = row[2]

        strong_boundaries.append(
            (direction, x, y, difference)
        )

print("")
print("Strong boundary count:", len(strong_boundaries))
print("")

for i, row in enumerate(strong_boundaries, start=1):

    direction = row[0]
    x = row[1]
    y = row[2]
    difference = row[3]

    print(
        i,
        direction,
        "X:", x,
        "Y:", y,
        "Difference:", round(difference, 3)
    )

# Group nearby strong boundaries

distance_limit = roi_size * 1.5

groups = []

for boundary in strong_boundaries:

    direction = boundary[0]
    x = boundary[1]
    y = boundary[2]
    difference = boundary[3]

    added = False

    for group in groups:

        for existing in group:

            ex_direction = existing[0]
            ex_x = existing[1]
            ex_y = existing[2]

            if direction != ex_direction:
                continue

            distance = (
                (x - ex_x) ** 2 +
                (y - ex_y) ** 2
            ) ** 0.5

            if distance <= distance_limit:

                group.append(boundary)
                added = True
                break

        if added:
            break

    if not added:

        groups.append([boundary])


print("")
print("Boundary groups:", len(groups))
print("")

for i, group in enumerate(groups, start=1):

    print("Group", i)

    for boundary in group:

        print(
            " ",
            boundary[0],
            "X:", boundary[1],
            "Y:", boundary[2],
            "Difference:", round(boundary[3], 3)
        )

# Check diagonal continuity of strong boundaries

print("")
print("Boundary continuity")
print("")

for i, boundary_a in enumerate(strong_boundaries):

    direction_a = boundary_a[0]
    x_a = boundary_a[1]
    y_a = boundary_a[2]

    for j, boundary_b in enumerate(strong_boundaries):

        if i >= j:
            continue

        direction_b = boundary_b[0]
        x_b = boundary_b[1]
        y_b = boundary_b[2]

        if direction_a != direction_b:
            continue

        dx = x_b - x_a
        dy = y_b - y_a

        distance = (
            dx ** 2 +
            dy ** 2
        ) ** 0.5

        if distance <= roi_size * 3:

            print(
                "Boundary",
                i + 1,
                "<->",
                j + 1,
                "Distance:",
                round(distance, 1),
                "dx:", dx,
                "dy:", dy
            )

# Create a boundary map

boundary_map = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

for row in strong_boundaries:

    direction = row[0]
    x = row[1]
    y = row[2]

    if direction == "Vertical":

        cv2.line(
            boundary_map,
            (x, y),
            (x, y + roi_size),
            (0, 0, 255),
            4
        )

    else:

        cv2.line(
            boundary_map,
            (x, y),
            (x + roi_size, y),
            (0, 0, 255),
            4
        )

display_width = 900

scale = display_width / width

display_height = int(
    height * scale
)

boundary_map = cv2.resize(
    boundary_map,
    (
        display_width,
        display_height
    )
)

cv2.imshow(
    "Boundary Map",
    boundary_map
)

print("")
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

# Display boundaries with lower threshold

threshold = 3

boundary_display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

candidate_number = 1

for row in boundary_strengths:

    difference = row[3]

    if difference >= threshold:

        direction = row[0]
        x = row[1]
        y = row[2]

        if direction == "Vertical":

            cv2.line(
                boundary_display,
                (x, y),
                (x, y + roi_size),
                (0, 0, 255),
                3
            )

            cv2.putText(
                boundary_display,
                str(candidate_number),
                (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        else:

            cv2.line(
                boundary_display,
                (x, y),
                (x + roi_size, y),
                (0, 0, 255),
                3
            )

            cv2.putText(
                boundary_display,
                str(candidate_number),
                (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        candidate_number += 1

display_width = 900

scale = display_width / width

display_height = int(
    height * scale
)

boundary_display = cv2.resize(
    boundary_display,
    (
        display_width,
        display_height
    )
)

cv2.imshow(
    "Boundary Candidates Threshold 3",
    boundary_display
)

print("")
print("Threshold:", threshold)
print("Candidate count:", candidate_number - 1)
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

# Display Canny edges

blurred = cv2.GaussianBlur(
    display_img,
    (5, 5),
    0
)

edges = cv2.Canny(
    blurred,
    20,
    60
)

display_width = 900

scale = display_width / width

display_height = int(
    height * scale
)

edges_display = cv2.resize(
    edges,
    (
        display_width,
        display_height
    )
)

cv2.imshow(
    "Canny Edges",
    edges_display
)

print("")
print("Canny edge detection")
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

# Refine strong boundaries using local intensity change
#
# For each strong boundary, search within +/- 20 pixels.
# Instead of taking the nearest Canny edge, select the position
# with the largest local intensity change.
# The original K-means and boundary detection are unchanged.

refined_boundary_display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

refine_tolerance = 20
refined_point_count = 0

for row in strong_boundaries:

    direction = row[0]
    x = row[1]
    y = row[2]

    if direction == "Vertical":

        for yy in range(y, min(y + roi_size, height)):

            best_x = None
            best_change = -1

            for offset in range(
                -refine_tolerance,
                refine_tolerance + 1
            ):

                xx = x + offset

                if 1 <= xx < width - 1:

                    local_change = abs(
                        int(img[yy, xx + 1]) -
                        int(img[yy, xx - 1])
                    )

                    if local_change > best_change:

                        best_change = local_change
                        best_x = xx

            if best_x is not None and best_change > 0:

                cv2.circle(
                    refined_boundary_display,
                    (best_x, yy),
                    1,
                    (0, 0, 255),
                    -1
                )

                refined_point_count += 1

    else:

        for xx in range(x, min(x + roi_size, width)):

            best_y = None
            best_change = -1

            for offset in range(
                -refine_tolerance,
                refine_tolerance + 1
            ):

                yy = y + offset

                if 1 <= yy < height - 1:

                    local_change = abs(
                        int(img[yy + 1, xx]) -
                        int(img[yy - 1, xx])
                    )

                    if local_change > best_change:

                        best_change = local_change
                        best_y = yy

            if best_y is not None and best_change > 0:

                cv2.circle(
                    refined_boundary_display,
                    (xx, best_y),
                    1,
                    (0, 0, 255),
                    -1
                )

                refined_point_count += 1

display_width = 900

scale = display_width / width

display_height = int(
    height * scale
)

refined_boundary_display = cv2.resize(
    refined_boundary_display,
    (
        display_width,
        display_height
    ),
    interpolation=cv2.INTER_NEAREST
)

cv2.imshow(
    "Refined Boundary Points",
    refined_boundary_display
)

print("")
print("Refined boundary points:", refined_point_count)
print("Red points = strongest local intensity changes near strong boundaries")
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

# Whole-image boundary tracing with a coarse lung-field exclusion mask
#
# main_30 is retained as the base detector.
# Instead of trying to estimate the lung field from image intensity,
# use a coarse anatomical mask only as a visualization filter.
#
# Goal:
#   - suppress edges well inside the lung fields
#   - retain the lung-field boundary
#   - retain strong contours outside the lung fields
#
# This is a visualization heuristic, not diagnostic segmentation.

trace_blur = cv2.GaussianBlur(
    display_img,
    (5, 5),
    0
)

trace_edges = cv2.Canny(
    trace_blur,
    10,
    30
)

# ------------------------------------------------------------
# Coarse lung-field polygons
# ------------------------------------------------------------
# These polygons deliberately describe only the broad interior
# of the two lung fields.  They are not intended to be exact
# anatomical segmentation.

left_lung_polygon = np.array([
    (int(width * 0.39), int(height * 0.13)),
    (int(width * 0.31), int(height * 0.16)),
    (int(width * 0.24), int(height * 0.25)),
    (int(width * 0.19), int(height * 0.40)),
    (int(width * 0.18), int(height * 0.62)),
    (int(width * 0.20), int(height * 0.78)),
    (int(width * 0.25), int(height * 0.87)),
    (int(width * 0.35), int(height * 0.90)),
    (int(width * 0.44), int(height * 0.82)),
    (int(width * 0.48), int(height * 0.68)),
    (int(width * 0.48), int(height * 0.30))
], dtype=np.int32)

right_lung_polygon = np.array([
    (int(width * 0.61), int(height * 0.13)),
    (int(width * 0.69), int(height * 0.16)),
    (int(width * 0.76), int(height * 0.25)),
    (int(width * 0.81), int(height * 0.40)),
    (int(width * 0.82), int(height * 0.62)),
    (int(width * 0.80), int(height * 0.78)),
    (int(width * 0.75), int(height * 0.87)),
    (int(width * 0.65), int(height * 0.90)),
    (int(width * 0.56), int(height * 0.82)),
    (int(width * 0.52), int(height * 0.68)),
    (int(width * 0.52), int(height * 0.30))
], dtype=np.int32)

lung_interior = np.zeros(
    (height, width),
    dtype=np.uint8
)

cv2.fillPoly(
    lung_interior,
    [left_lung_polygon],
    255
)

cv2.fillPoly(
    lung_interior,
    [right_lung_polygon],
    255
)

# Keep a narrow band around the estimated lung boundary.
boundary_band = cv2.dilate(
    lung_interior,
    np.ones((31, 31), np.uint8),
    iterations=1
)

# Edges far inside the lung fields are suppressed.
# Edges outside the polygons and edges near their border remain.
far_inside = (
    (lung_interior > 0) &
    (boundary_band == 0)
)

filtered_edges = trace_edges.copy()

filtered_edges[far_inside] = 0

# Small gaps are closed without changing the basic edge structure.
trace_kernel = np.ones(
    (3, 3),
    np.uint8
)

filtered_edges = cv2.morphologyEx(
    filtered_edges,
    cv2.MORPH_CLOSE,
    trace_kernel,
    iterations=1
)

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        filtered_edges,
        connectivity=8
    )
)

whole_image_boundary_display = cv2.cvtColor(
    display_img,
    cv2.COLOR_GRAY2BGR
)

traced_components = 0

for label in range(1, num_labels):

    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]
    area = stats[label, cv2.CC_STAT_AREA]

    long_horizontal = w >= 80
    long_vertical = h >= 80
    sufficiently_large = area >= 20

    if sufficiently_large and (
        long_horizontal or long_vertical
    ):

        component_mask = (
            labels[y:y + h, x:x + w] == label
        ).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        for contour in contours:

            if len(contour) < 15:
                continue

            contour = contour.copy()

            contour[:, 0, 0] += x
            contour[:, 0, 1] += y

            cv2.drawContours(
                whole_image_boundary_display,
                [contour],
                -1,
                (0, 0, 255),
                2
            )

        traced_components += 1

display_width = 900

scale = display_width / width

display_height = int(
    height * scale
)

whole_image_boundary_display = cv2.resize(
    whole_image_boundary_display,
    (
        display_width,
        display_height
    ),
    interpolation=cv2.INTER_NEAREST
)

cv2.imshow(
    "Whole Image Boundary Trace",
    whole_image_boundary_display
)

print("")
print("Filtered boundary components:", traced_components)
print("Red lines = strong contours outside/near the lung-field boundary")
print("Internal lung-field edges are suppressed where they are well inside the coarse mask.")
print("This is a visualization filter, not a diagnostic segmentation.")
print("Press any key on the image window.")

cv2.waitKey(0)

cv2.destroyAllWindows()

# Measure Canny edge overlap

edge_tolerance = 5

print("")
print("Canny Edge Overlap")
print("")

for i, row in enumerate(strong_boundaries, start=1):

    direction = row[0]
    x = row[1]
    y = row[2]

    edge_count = 0
    total_count = 0

    if direction == "Vertical":

        for yy in range(y, y + roi_size):

            found_edge = False

            for offset in range(
                -edge_tolerance,
                edge_tolerance + 1
            ):

                xx = x + offset

                if 0 <= xx < width:

                    if edges[yy, xx] > 0:

                        found_edge = True
                        break

            if found_edge:

                edge_count += 1

            total_count += 1

    else:

        for xx in range(x, x + roi_size):

            found_edge = False

            for offset in range(
                -edge_tolerance,
                edge_tolerance + 1
            ):

                yy = y + offset

                if 0 <= yy < height:

                    if edges[yy, xx] > 0:

                        found_edge = True
                        break

            if found_edge:

                edge_count += 1

            total_count += 1

    edge_ratio = edge_count / total_count

    print(
        i,
        direction,
        "X:", x,
        "Y:", y,
        "Edge ratio:",
        round(edge_ratio, 3)
    )

# Measure Canny edge overlap with wider tolerance

edge_tolerance = 20

print("")
print("Canny Edge Overlap")
print("")

for i, row in enumerate(strong_boundaries, start=1):

    direction = row[0]
    x = row[1]
    y = row[2]

    edge_count = 0
    total_count = 0

    if direction == "Vertical":

        for yy in range(y, y + roi_size):

            found_edge = False

            for offset in range(
                -edge_tolerance,
                edge_tolerance + 1
            ):

                xx = x + offset

                if 0 <= xx < width:

                    if edges[yy, xx] > 0:

                        found_edge = True
                        break

            if found_edge:
                edge_count += 1

            total_count += 1

    else:

        for xx in range(x, x + roi_size):

            found_edge = False

            for offset in range(
                -edge_tolerance,
                edge_tolerance + 1
            ):

                yy = y + offset

                if 0 <= yy < height:

                    if edges[yy, xx] > 0:

                        found_edge = True
                        break

            if found_edge:
                edge_count += 1

            total_count += 1

    edge_ratio = edge_count / total_count

    print(
        i,
        direction,
        "X:", x,
        "Y:", y,
        "Edge ratio:",
        round(edge_ratio, 3)
    )



rows = height // roi_size
cols = width // roi_size

continuous_boundaries = []

for row in boundary_strengths:

    direction = row[0]
    x = row[1]
    y = row[2]
    difference = row[3]

    if difference < 3:
        continue

    if direction == "Horizontal":

        current_col = x // roi_size
        current_row = y // roi_size

        for other in boundary_strengths:

            if other[0] != "Horizontal":
                continue

            if other[3] < 3:
                continue

            other_col = other[1] // roi_size
            other_row = other[2] // roi_size

            if other_row == current_row:

                if abs(other_col - current_col) == 1:

                    continuous_boundaries.append(
                        (
                            x,
                            y,
                            other[1],
                            other[2]
                        )
                    )

    else:

        current_col = x // roi_size
        current_row = y // roi_size

        for other in boundary_strengths:

            if other[0] != "Vertical":
                continue

            if other[3] < 3:
                continue

            other_col = other[1] // roi_size
            other_row = other[2] // roi_size

            if other_col == current_col:

                if abs(other_row - current_row) == 1:

                    continuous_boundaries.append(
                        (
                            x,
                            y,
                            other[1],
                            other[2]
                        )
                    )


print("")
print("Continuous Boundary Pairs:", len(continuous_boundaries))

for pair in continuous_boundaries:

    print(
        "Boundary:",
        pair[0],
        pair[1],
        "<->",
        pair[2],
        pair[3]
    )

# ============================================================
# Check group consistency around strong boundaries
# ============================================================

print("")
print("Group consistency around strong boundaries")
print("")

for i, boundary in enumerate(strong_boundaries, start=1):

    direction = boundary[0]
    x = boundary[1]
    y = boundary[2]

    col = x // roi_size
    row = y // roi_size

    if direction == "Horizontal":

        if row == 0:
            continue

        group_a = labels_grid[row - 1, col]
        group_b = labels_grid[row, col]

    else:

        if col == 0:
            continue

        group_a = labels_grid[row, col - 1]
        group_b = labels_grid[row, col]

    print(
        i,
        direction,
        "X:", x,
        "Y:", y,
        "Group:",
        group_a + 1,
        "<->",
        group_b + 1
    )