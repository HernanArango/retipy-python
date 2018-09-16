# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Module with operations related to detect and classify crossings and bifurcations."""

import numpy as np
import cv2
from retipy import retina


def potential_landmarks(image: np.ndarray, kernel: int):
    binary = image.copy()
    binary[binary == 255] = 1
    result = image.copy()
    n = int(np.floor(kernel / 2))
    potential = []
    for it_x in range(n, image.shape[0] - n):
        for it_y in range(n, image.shape[1] - n):
            aux = 0
            if binary[it_x, it_y] == 1:
                aux += np.sum(binary[it_x - n:it_x + n + 1, it_y - n:it_y + n + 1])
                if aux == 4:
                    result[it_x, it_y] = 0
                    potential.append([it_x, it_y])
                elif aux >= 5:
                    result[it_x, it_y] = 0
                    potential.append([it_x, it_y])
    return potential


def vessel_width(thresholded_image: np.ndarray, landmarks: list):
    image = thresholded_image.copy()
    widths = []
    for j in landmarks:
        w0 = w45 = w90 = w135 = w180 = w225 = w270 = w315 = 1
        while True:
            if image[j[0], j[1] + w0] != 0:
                w0 += 1
            if image[j[0], j[1] - w180] != 0:
                w180 += 1
            if image[j[0] - w45, j[1] + w45] != 0:
                w45 += 1
            if image[j[0] + w225, j[1] - w225] != 0:
                w225 += 1
            if image[j[0] - w90, j[1]] != 0:
                w90 += 1
            if image[j[0] + w270, j[1]] != 0:
                w270 += 1
            if image[j[0] - w135, j[1] - w135] != 0:
                w135 += 1
            if image[j[0] + w315, j[1] + w315] != 0:
                w315 += 1

            if image[j[0], j[1] + w0] == 0 and image[j[0], j[1] - w180] == 0 and (w0 > 1 or w180 > 1):
                widths.append([0, w0, w180 - 1])
                break
            elif image[j[0] - w45, j[1] + w45] == 0 and image[j[0] + w225, j[1] - w225] == 0 and (w45 > 1 or w225 > 1):
                widths.append([45, w45, w225 - 1])
                break
            elif image[j[0] - w90, j[1]] == 0 and image[j[0] + w270, j[1]] == 0 and (w90 > 1 or w270 > 1):
                widths.append([90, w90, w270 - 1])
                break
            elif image[j[0] - w135, j[1] - w135] == 0 and image[j[0] + w315, j[1] + w315] == 0 and (
                    w135 > 1 or w315 > 1):
                widths.append([135, w135, w315 - 1])
                break

    return widths


def finding_landmark_vessels(widths: list, landmarks: list, skeleton: np.ndarray, skeleton_rgb: np.ndarray):
    vessels = []
    for l in range(0, len(widths)):
        cgray = skeleton.copy()
        crgb = skeleton_rgb.copy()
        radius = int(np.ceil(widths[l][1] + widths[l][2] * 1.4))
        x0 = landmarks[l][0]
        y0 = landmarks[l][1]
        points = []
        dy = x = y = 0
        crgb[x0, y0] = [0, 255, 0]
        for start in range(0, 2):
            for rad in range(-radius, radius + 1):
                dy = int(np.round(np.sqrt(np.power(radius, 2) - np.power(rad, 2))))
                for loop in range(0, 2):
                    if start == 0:
                        x = x0 + rad
                        if loop == 0:
                            y = y0 - dy
                        else:
                            y = y0 + dy
                    else:
                        y = y0 + rad
                        if loop == 0:
                            x = x0 - dy
                        else:
                            x = x0 + dy

                    acum = 0
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            if all(crgb[x + i, y + j] == [0, 0, 255]):
                                acum += 1

                    if cgray[x, y] == 255 and acum == 0:
                        crgb[x, y] = [0, 0, 255]
                        cgray[x - 1:x + 2, y - 1:y + 2] = 0
                        cgray[x, y] = 255
                        points.append([x, y])
                    elif acum == 0:
                        crgb[x, y] = [255, 0, 0]
                        block = cgray[x - 1:x + 2, y - 1:y + 2]
                        connected_components = cv2.connectedComponentsWithStats(block.astype(np.uint8), 8, cv2.CV_8U)
                        for k in range(1, connected_components[0]):
                            mask = connected_components[1] == k
                            indexes = np.column_stack(np.where(mask))
                            for e in range(0, len(indexes)):
                                ix = x + indexes[e][0] - 1
                                iy = y + indexes[e][1] - 1
                                if e == int(len(indexes) / 2):
                                    crgb[ix, iy] = [0, 0, 255]
                                    points.append([ix, iy])

        vessels.append(points)
    return vessels


def vessel_number(vessels: list, landmarks: list, skeleton_rgb: np.ndarray):
    skeleton = skeleton_rgb.copy()
    length = len(vessels)
    for v in range(0, length):
        if len(vessels[v]) == 3:
            skeleton[landmarks[v][0], landmarks[v][1]] = [0, 0, 255]
        elif len(vessels[v]) == 4:
            skeleton[landmarks[v][0], landmarks[v][1]] = [255, 0, 0]

    return skeleton


def boxes_auxiliary(skeleton: np.ndarray, landmarks: list, bifurcations_coordinates: list, crossings_coordinates: list):
    x = landmarks[0][0]
    y = landmarks[0][1]
    num_bifurcations = 0
    num_crossings = 0
    box = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            box.append([x + i, y + j])
            if all(skeleton[x + i, y + j] == [0, 0, 255]):
                num_bifurcations += 1
            elif all(skeleton[x + i, y + j] == [255, 0, 0]):
                num_crossings += 1

    landmarks = [val for val in landmarks if val not in box]

    if num_bifurcations > num_crossings:
        bifurcations_coordinates.append([y - 3, x - 3, y + 3, x + 3])
    else:
        crossings_coordinates.append([y - 3, x - 3, y + 3, x + 3])

    return landmarks


def principal_boxes(skeleton: np.ndarray, landmarks: list):
    junct = landmarks.copy()
    bifurcations_coordinates = []
    crossings_coordinates = []
    while True:
        if junct:
            junct = boxes_auxiliary(skeleton, junct, bifurcations_coordinates, crossings_coordinates)
        else:
            break

    return bifurcations_coordinates, crossings_coordinates


def classification(image: np.ndarray):
    img = retina.Retina(image, None)
    img.threshold_image()
    threshold = img.get_uint_image()
    img.skeletonization()
    skeleton = img.get_uint_image()
    img.bin_to_bgr()
    skeleton_rgb = img.get_uint_image()

    landmarks = potential_landmarks(skeleton, 3)
    widths = vessel_width(threshold, landmarks)
    vessels = finding_landmark_vessels(widths, landmarks, skeleton, skeleton_rgb)
    marked_skeleton = vessel_number(vessels, landmarks, skeleton_rgb)
    bifurcations, crossings = principal_boxes(marked_skeleton, landmarks)

    return bifurcations, crossings