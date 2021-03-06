import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import imutils



template1 = "./retipy/retipy/templates_od/od1.png"
template2 = "./retipy/retipy/templates_od/od2.png"
template3 = "./retipy/retipy/templates_od/od3.png"
template4 = "./retipy/retipy/templates_od/od4.png"
"""
#test
template1 = "../retipy/templates_od/od1.png"
template2 = "../retipy/templates_od/od2.png"
template3 = "../retipy/templates_od/od3.png"
template4 = "../retipy/templates_od/od4.png"
"""

classification_scale = {"Normal": 0, "Medium": 0,"Large": 0}

def show_image(image, tittle):
    pass
    #cv2.imshow(tittle, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def removing_dark_pixel(image):
    b, g, r = cv2.split(image)
    rows, cols,_ = image.shape
    new_matriz = np.zeros((rows, cols))
    average = np.average(r)
    max = np.amax(image)

    total = 0
    for i in range(0, rows):
        for j in range(0, cols):
            total += image[i][j][0]
            if r[i][j] <= average:
                image[i][j][0] = 0
                image[i][j][1] = 0
                image[i][j][2] = 0
    return image


def threshold(img, t):
    rows, cols = img.shape
    new_matriz = np.zeros((rows, cols))
    # creating binary matrix
    for i in range(0, rows):
        for j in range(0, cols):
            if img[i][j] >= t:
                new_matriz[i][j] = 1
            else:
                new_matriz[i][j] = 0
    return new_matriz


def detect_roi(img, optic_disc):
    rows, cols, _ = img.shape
    x = optic_disc[0]
    y = optic_disc[1]
    # distance of optic disc to the macula
    distance = int(cols*0.35)
    middle_image = int(cols/2)
    """
    cv2.circle(img, (x, y), 0, (255, 0, 0), 40)
    font = cv2.FONT_ITALIC
    cv2.putText(img, 'Disco Optico', (x+20,y+15), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
    """
    translate_rect = 0
    # detect if the macula is to the left or right
    #right
    if x > middle_image:

        x = x - distance
        translate_rect = 0
    #left
    else:

        x = x + distance
        translate_rect = 0
    """
    cv2.circle(img, (x, y), 0, (0, 255, 0), 40)
    cv2.putText(img, 'Fovea', (x+20,y+15), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
    """


    width_roi = cols * 0.387
    height_roi = rows * 0.51

    original_image = copy.copy(img)

    cv2.rectangle(original_image, (x - 500 +translate_rect, y - 450), (x + 500 + translate_rect, y + 550), (0, 255, 0), 3)
    show_image(original_image,"roi1")
    #cv2.rectangle(img, (x - round(width_roi/2) +translate_rect, y - round(height_roi/2)), (x + round(width_roi/2) + translate_rect, y + round(height_roi/2)), (0, 255, 0), 3)
    #show_image(imutils.resize(img, width=700),"roi")



    # print(x - 200+translate_rect)
    roi = img[y - round(height_roi/2):y + round(height_roi/2), (x - round(width_roi/2)):(x + round(width_roi/2))]
    #roi = img[y - 450:y + 550, (x - 500)+translate_rect:(x + 500)+translate_rect]


    return roi

def get_mask(img):
    b, g, r = cv2.split(img)
    th = threshold(r, 35)
    kernel = np.ones((3, 3), np.uint8)
    # openning erosion and dilatation
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(closing, kernel, iterations=1)
    result = apply_mask(img, mask)
    return result


def apply_mask(img, mask):
    rows, cols, _ = img.shape

    for i in range(0, rows):
        for j in range(0, cols):
            if mask[i][j] == 0:
                img[i][j][0] = 0
                img[i][j][1] = 0
                img[i][j][2] = 0
    return img


def aux_template_optic_disc(template):
    image = cv2.imread(template)
    blue_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    red_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    return [blue_hist, green_hist, red_hist]


def template_optic_disc():
    hist1 = aux_template_optic_disc(template1)
    hist2 = aux_template_optic_disc(template2)
    hist3 = aux_template_optic_disc(template3)
    hist4 = aux_template_optic_disc(template4)

    blue_hist = []
    for i in range(0, 256):
        blue_hist.append((hist1[0][i] + hist2[0][i] + hist3[0][i] + hist4[0][i]) / 4)

    green_hist = []
    for i in range(0, 256):
        green_hist.append((hist1[1][i] + hist2[1][i] + hist3[1][i] + hist4[1][i]) / 4)

    red_hist = []
    for i in range(0, 256):
        red_hist.append((hist1[2][i] + hist2[2][i] + hist3[2][i] + hist4[2][i]) / 4)

    return [blue_hist, green_hist, red_hist]


def hist_window(image, y, x):
    # window 80x80
    img_window = image[(y - 40):(y + 40), (x - 40):(x + 40)]
    blue_hist = cv2.calcHist([img_window], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([img_window], [1], None, [256], [0, 256])
    red_hist = cv2.calcHist([img_window], [2], None, [256], [0, 256])
    return [blue_hist, green_hist, red_hist]


def hist_correlation(templates_histograms, histograms_window):
    difference_histograms_b = np.sum(pow(templates_histograms[0] - histograms_window[0], 2))
    difference_histograms_g = np.sum(pow(templates_histograms[1] - histograms_window[1], 2))
    difference_histograms_r = np.sum(pow(templates_histograms[2] - histograms_window[2], 2))

    Cb = 1 / (1 + difference_histograms_b)
    Cg = 1 / (1 + difference_histograms_g)
    Cr = 1 / (1 + difference_histograms_r)
    Tb = 1
    Tg = 2
    Tr = 0.5

    C = (Tr * Cr) + (Tb * Cb) + (Tg * Cg)

    return C


def threshold_color(otsu_img,img):
    b, g, r = cv2.split(img)
    average = np.average(g)
    rows, cols = otsu_img.shape

    for i in range(0, rows):
        for j in range(0, cols):

            if otsu_img[i][j] != 0:
                if g[i][j] < average:
                    otsu_img[i][j] = 0

    return otsu_img

def non_uniform_ilumination_correction(img):
    b, g, r = cv2.split(img)
    #show_image(imutils.resize(g, width=700),"fin1")
    fundus = cv2.medianBlur(g, 71)
    new_image = (g/fundus)
    #new_image = (new_image*0.2)
    new_image = (new_image*55).astype(np.uint8)
    show_image(imutils.resize(new_image, width=700),"final")
    new_image = cv2.GaussianBlur(new_image,(1,1),0,0,cv2.BORDER_DEFAULT)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    #resaltada = clahe.apply(new_image)
    #show_image(imutils.resize(resaltada, width=700),"final resaltada")
    return new_image


def detect_drusen(img):

    b, g, r = cv2.split(img)

    g = non_uniform_ilumination_correction(img)
    #fundus = cv2.medianBlur(g, 41)
    fundus = cv2.medianBlur(g, 71)
    #g = cv2.GaussianBlur(g,(7,7),0,0,cv2.BORDER_DEFAULT)
    g = cv2.GaussianBlur(g,(15,15),0,0,cv2.BORDER_DEFAULT)
    #g3 = cv2.GaussianBlur(g,(27,27),0,0,cv2.BORDER_DEFAULT)

    #show_image(imutils.resize(g, width=700),"green")
    #show_image(imutils.resize(g2, width=700),"green2")
    #show_image(imutils.resize(g3, width=700),"green3")

    #show_image(imutils.resize(g, width=700),"g blur")
    #show_image(imutils.resize(fundus, width=700),"fundus")
    #x = (fundus/g3)*1.09
    x = (fundus/g)*1.09
    #show_image(imutils.resize(x, width=700),"x")
    new_image = (x*255).astype(np.uint8)
    #show_image(imutils.resize(new_image, width=700),"otsu n")
    # threshold Otsu
    ret, otsu_img = cv2.threshold(new_image, 0, 255, cv2.THRESH_OTSU)
    #show_image(imutils.resize(otsu_img, width=700),"con venas")
    veins = detect_veins(g)
    # otsu_img = (otsu_img/veins).astype(np.uint8)


    # evite division by zero
    edge_map = np.zeros_like(veins)
    non_zero = veins != 0
    edge_map[non_zero] = otsu_img[non_zero]/veins[non_zero]
    otsu_img = edge_map



    ret, otsu_img = cv2.threshold(otsu_img, 0, 255, cv2.THRESH_OTSU)
    #show_image(otsu_img,"segmentaicon")

    otsu_img = threshold_color(otsu_img,img)
    show_image(otsu_img,"segmentaicon2")

    #using erotion and dilation for evite count more contours by error
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    otsu_img = cv2.morphologyEx(otsu_img, cv2.MORPH_OPEN, kernel)

    contours,_ = cv2.findContours(otsu_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print("contornos",len(contours))

    # draw contours

    for c in contours:

        momentos = cv2.moments(c)
        if momentos['m10']== 0 or momentos['m00']==0:
            cx = 0
        else:
            cx = int(momentos['m10']/momentos['m00'])


        if momentos['m01']== 0 or momentos['m00']==0:
            cy = 0
        else:
            cy = int(momentos['m01']/momentos['m00'])

        #cv2.circle(img,(cx, cy), 3, (0,0,255), -1)

        #size_drusen([cx,cy],momentos)
        #break

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        #im = cv2.drawContours(img,[box],0,(255,0,0),2)
        #show_image(imutils.resize(im, width=700),"contornos rect")
        # calculate the size drusen and return the size classification scale
        classification = size_drusen(rect[1])

        # Normal drusen
        if classification == 1:
            # draw contours
            cv2.drawContours(img, [c], 0, (0, 255, 0), 2, cv2.LINE_AA)
        # Medium Drusen
        elif classification == 2:
            # draw contours
            cv2.drawContours(img, [c], 0, (255, 0, 0), 2, cv2.LINE_AA)
        # Large Drusen
        else:
            # draw contours
            cv2.drawContours(img, [c], 0, (0, 0, 255), 2, cv2.LINE_AA)


        # draw contours


        """
        cx = int(momentos['m10']/momentos['m00'])
        cy = int(momentos['m01']/momentos['m00'])
        #Dibujar el centro
        cv2.circle(img,(cx, cy), 3, (0,0,255), -1)
        """


    show_image(imutils.resize(img, width=700),"contornos")
    return img




def size_drusen(dimensions):
    diameter = 0
    #bigest side
    if dimensions[0] > dimensions[1]:
        diameter = dimensions[0]
    else:
        diameter = dimensions[1]

    micron = 3.4
    # transform diameter in pixel to micron -> 1px = 3.4 micron
    diameter = diameter * micron

    scale = 0
    #normal --->   <= 63 micron
    if diameter <= 63:
        classification_scale["Normal"] += 1
        scale = 1
    #Early AMD --->  Medium Drusen > 63 micron and <= 125 miron
    elif diameter > 63 and diameter <= 125:
        classification_scale["Medium"] += 1
        scale = 2
    #Intermetiate AMD --> Large Drusen > 125 micron
    else:
        classification_scale["Large"] += 1
        scale = 3

    return scale


def detect_optical_disc(image):
    start = time.time()
    # print("mask")
    original_image = copy.copy(image)
    #image = get_mask(image)
    # show_image(image, "normal")
    # print("template")
    templates_histograms = template_optic_disc()

    # average = np.average(image)
    # print("average",average)

    b, g, r = cv2.split(image)
    image = cv2.medianBlur(image, 5)

    rows, cols, _ = image.shape

    # window_histogram = hist_window(image,500,200)
    # hist_correlation(templates_histograms,window_histogram)

    correlations = []
    new_matriz = np.zeros((rows, cols))
    # print(rows, cols)
    for i in range(200, rows-230):
        # print(i)
        for j in range(0, cols):


            if image[i][j][0] != 0:
                window_histogram = hist_window(image, i, j)
                new_matriz[i][j] = hist_correlation(templates_histograms, window_histogram)

    # print("calculando el max")
    max = new_matriz[0][0]
    for i in range(0, rows):
        for j in range(0, cols):
            if new_matriz[i][j] > max:
                max = new_matriz[i][j]

    # print("max", max)
    # print("threshold")
    image_threshold = threshold(new_matriz, max * 0.7)
    end = time.time()
    # print(end - start)
    kernel = np.ones((3, 3), np.uint8)
    # show_image(image_threshold, "image threshold")
    # opening = erosion followed by dilation
    gradient = cv2.morphologyEx(image_threshold, cv2.MORPH_GRADIENT, kernel)
    # show_image(gradient, "gradient")

    # calulating the center of optic disc
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0
    for i in range(0, rows):
        for j in range(0, cols):
            # only enter one time
            if gradient[i][j] == 1 and y_max == 0:
                # initial definition only one time enter here
                y_min = i
                x_min = j
                x_max = j
            if gradient[i][j] == 1:
                y_max = i

                if x_min > j:
                    x_min = j
                elif x_max < j:
                    x_max = j

    # print(x_min, x_max, y_min, y_max)
    y = int(y_min + ((y_max - y_min) / 2))
    x = int(x_min + ((x_max - x_min) / 2))

    # x = 142
    # y = 227
    #cv2.circle(image, (x, y), 2, (255, 0, 0), 3)
    ## show_image(image, 'circles')

    return [x,y]


def detect_veins(g):


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    gray = clahe.apply(g)
    image_filtered = cv2.GaussianBlur(gray,(21,21),0)
    th2 = cv2.adaptiveThreshold(image_filtered,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    # show_image(th2,"hola")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    dilate = cv2.dilate(th2, kernel)
    #dilate = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    #show_image(dilate,"veins")

    return dilate

def change_resolution(img):
    img = imutils.resize(img, width=700)
    return img



def main(image, debug = False):

    original_image = copy.copy(image)
    if debug:
        print("# Changing Resolution")
    image = change_resolution(image)



    cols_original, rows_original, _ = original_image.shape
    cols_modified, rows_modified, _ = image.shape
    # Get the original ratio
    Rx = (rows_original/rows_modified)
    Ry = (cols_original/cols_modified)

    if debug:
        print("# Removing Dark Pixels")
    image = removing_dark_pixel(image)
    if debug:
        print("# Detecting Optical Disc")
    x,y = detect_optical_disc(image)
    #x = 525
    #y = 206
    print(x,y)


    #cv2.circle(image, (x, y), 2, (255, 0, 0), 3)

    if debug:
        print("# Calculating ROI")
    roi = detect_roi(original_image, [round(x*Rx), round(y*Ry)])
    if debug:
        print("# Segmenting Drusen")
    print(roi.shape)
    drusen = detect_drusen(roi)


    return [drusen,classification_scale]
