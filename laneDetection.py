################################################################################
######## LANE DETECTION PROJECT ################################################
################################################################################
# BY:           CAN OZCIVELEK
# DATE:         DECEMBER 2018
#
# DESCRIPTION:  THIS  PROJECT WAS CREATED  TO DEMONSTRATE HOW  A  LANE DETECTION
#               SYSTEM WORKS  ON CARS EQUIPPED WITH A FRONT  FACING CAMERA. WITH
#               THE HELP OF OPENCV LIBRARIES IT IS POSSIBLE TO DESIGN ALGORITHMS
#               THAT CAN  IDENTIFY LANE LINES, AND PREDICT STEERING ANGLES, ALSO
#               WARN  DRIVERS  IF THE CAR IS  DRIFTING  AWAY FROM  CURRENT LANE.
################################################################################
import base64

# IMPORT NECESSARY LIBRARIES
import cv2 as cv
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors

# Defining variables to hold meter-to-pixel conversion
ym_per_pix = 0.8 / 360
# Standard lane width is 0.35 meters divided by lane width in pixels which is
# calculated to be approximately 360 pixels not to be confused with frame height
xm_per_pix = 0.35 / 360
wheel_position = 0

# Get path to the current working directory
CWD_PATH = os.getcwd()


################################################################################
######## START - FUNCTIONS TO PERFORM IMAGE PROCESSING #########################
################################################################################

################################################################################
#### START - FUNCTION TO READ AN INPUT IMAGE ###################################
def readVideo():
    # Read input video from current working directory
    inpImage = cv.VideoCapture(os.path.join(CWD_PATH, 'resources/curba+semne/test3.avi'))

    return inpImage


#### END - FUNCTION TO READ AN INPUT IMAGE #####################################
################################################################################


################################################################################
#### START - FUNCTION TO PROCESS IMAGE #########################################
def processImage(inpImage):
    # Apply HLS color filtering to filter out white lane lines
    hls = cv.cvtColor(inpImage, cv.COLOR_BGR2HLS)
    lower_white = np.array([0, 160, 10])
    upper_white = np.array([255, 255, 255])
    mask = cv.inRange(inpImage, lower_white, upper_white)
    hls_result = cv.bitwise_and(inpImage, inpImage, mask=mask)

    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv.cvtColor(hls_result, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    blur = cv.GaussianBlur(thresh, (3, 3), 0)
    canny = cv.Canny(blur, 40, 60)

    # Display the processed images
    # cv.imshow("Image", inpImage)
    # cv.imshow("HLS Filtered", hls_result)
    # cv.imshow("Grayscale", gray)
    # cv.imshow("Thresholded", thresh)
    # cv.imshow("Blurred", blur)
    # cv.imshow("Canny Edges", canny)
    # cv.waitKey(0)

    return inpImage, hls_result, gray, thresh, blur, canny


#### END - FUNCTION TO PROCESS IMAGE ###########################################
################################################################################


################################################################################
#### START - FUNCTION TO APPLY PERSPECTIVE WARP ################################
def perspectiveWarp(inpImage):
    # Get image size
    img_size = (inpImage.shape[1], inpImage.shape[0])

    # Perspective points to be warped
    src = np.float32([[140, 100],
                      [340, 100],
                      [0, 360],
                      [480, 360]])

    # Window to be shown
    dst = np.float32([[75, 0],
                      [450, 0],
                      [75, 355],
                      [450, 355]])


    # Matrix to warp the image for birdseye window
    matrix = cv.getPerspectiveTransform(src, dst)
    # Inverse matrix to unwarp the image for final window
    minv = cv.getPerspectiveTransform(dst, src)
    birdseye = cv.warpPerspective(inpImage, matrix, img_size)

    # Get the birdseye window dimensions
    height, width = birdseye.shape[:2]

    # Divide the birdseye view into 2 halves to separate left & right lanes
    birdseyeLeft = birdseye[0:height, 0:width // 2]
    birdseyeRight = birdseye[0:height, width // 2:width]

    # Display birdseye view image
    # cv.imshow("Birdseye", birdseye)
    # cv.imshow("Birdseye Left", birdseyeLeft)
    # cv.imshow("Birdseye Right", birdseyeRight)

    return birdseye, birdseyeLeft, birdseyeRight, minv


#### END - FUNCTION TO APPLY PERSPECTIVE WARP ##################################
################################################################################


################################################################################
#### START - FUNCTION TO PLOT THE HISTOGRAM OF WARPED IMAGE ####################
def plotHistogram(inpImage):
    histogram = np.sum(inpImage[inpImage.shape[0] // 2:, :], axis=0)

    midpoint = int(histogram.shape[0] / 2)
    leftxBase = np.argmax(histogram[:midpoint])
    rightxBase = np.argmax(histogram[midpoint:]) + midpoint

    plt.xlabel("Image X Coordinates")
    plt.ylabel("Number of White Pixels")

    # Return histogram and x-coordinates of left & right lanes to calculate
    # lane width in pixels
    return histogram, leftxBase, rightxBase


#### END - FUNCTION TO PLOT THE HISTOGRAM OF WARPED IMAGE ######################
################################################################################


################################################################################
#### START - APPLY SLIDING WINDOW METHOD TO DETECT CURVES ######################
def slide_window_search(binary_warped, histogram):
    # Find the start of left and right lane lines using histogram info
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # A total of 9 windows will be used
    nwindows = 9
    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    #### START - Loop to iterate through windows and search for lane lines #####
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    #### END - Loop to iterate through windows and search for lane lines #######

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Apply 2nd degree polynomial fit to fit curves
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)
    plt.plot(right_fitx)
    # plt.show()

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='pink')
    plt.xlim(0, 480)
    plt.ylim(360, 0)

    return ploty, left_fit, right_fit, ltx, rtx


#### END - APPLY SLIDING WINDOW METHOD TO DETECT CURVES ########################
################################################################################


################################################################################
#### START - APPLY GENERAL SEARCH METHOD TO DETECT CURVES ######################
def general_search(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ## VISUALIZATION ###########################################################

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv.fillPoly(window_img, [np.array(left_line_pts, dtype=int)], (0, 255, 0))
    cv.fillPoly(window_img, [np.array(right_line_pts, dtype=int)], (0, 255, 0))
    result = cv.addWeighted(out_img, 1, window_img, 0.3, 0)

    # plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='pink')
    plt.xlim(0, 480)
    plt.ylim(360, 0)

    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = rightx
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = right_fitx
    ret['ploty'] = ploty

    return ret


#### END - APPLY GENERAL SEARCH METHOD TO DETECT CURVES ########################
################################################################################


################################################################################
#### START - FUNCTION TO MEASURE CURVE RADIUS ##################################
def measure_lane_curvature(ploty, leftx, rightx):
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    # Decide if it is a left or a right curve
    if leftx[0] - leftx[-1] > 60:
        curve_direction = 'Left Curve'
    elif leftx[-1] - leftx[0] > 60:
        curve_direction = 'Right Curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad + right_curverad) / 2.0, curve_direction


#### END - FUNCTION TO MEASURE CURVE RADIUS ####################################
################################################################################


################################################################################
#### START - FUNCTION TO VISUALLY SHOW DETECTED LANES AREA #####################
def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    leftx = draw_info['leftx']
    rightx = draw_info['rightx']
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv.fillPoly(color_warp, [np.array(pts, dtype=int)], (0, 255, 0))
    cv.fillPoly(color_warp, [np.array(pts_mean, dtype=int)], (0, 255, 0))

    newwarp = cv.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv.addWeighted(original_image, 1, newwarp, 0.3, 0)

    return pts_mean, result


#### END - FUNCTION TO VISUALLY SHOW DETECTED LANES AREA #######################
################################################################################


#### START - FUNCTION TO CALCULATE DEVIATION FROM LANE CENTER ##################
################################################################################
def offCenter(meanPts, inpFrame):
    # Calculating deviation in meters
    mpts = meanPts[-1][-1][-2].astype(int)
    pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
    deviation = pixelDeviation * xm_per_pix
    direction = "left" if deviation < 0 else "right"

    return deviation, direction


################################################################################
#### END - FUNCTION TO CALCULATE DEVIATION FROM LANE CENTER ####################


################################################################################
#### START - FUNCTION TO ADD INFO TEXT TO FINAL IMAGE ##########################
def addText(img,
            # radius, direction,
            # deviation, devDirection,
            current_wheel_position):
    # Add the radius and center position to the image
    font = cv.FONT_HERSHEY_TRIPLEX
    text2 = "Wheel position: " + str(current_wheel_position)

    # if (direction != 'Straight'):
    #     text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius) + 'm'
    #     text1 = 'Curve Direction: ' + (direction)
    #
    # else:
    #     text = 'Radius of Curvature: ' + 'N/A'
    #     text1 = 'Curve Direction: ' + (direction)
    #
    # cv.putText(img, text, (50, 25), font, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    # cv.putText(img, text1, (50, 75), font, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(img, text2, (50, 25), font, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    # Deviation
    # deviation_text = 'Off Center: ' + str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
    # cv.putText(img, deviation_text, (50, 175), cv.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    return img


#### END - FUNCTION TO ADD INFO TEXT TO FINAL IMAGE ############################
################################################################################

################################################################################
######## END - FUNCTIONS TO PERFORM IMAGE PROCESSING ###########################
################################################################################

################################################################################
################################################################################
################################################################################
################################################################################

def plotHistogramLeft(inpImageL):
    histogramL = np.sum(inpImageL[inpImageL.shape[0]:, :], axis=0)

    midpoint = int(histogramL.shape[0])
    leftxBase = np.argmax(histogramL[:midpoint])

    plt.xlabel("Image X Coordinates")
    plt.ylabel("Number of White Pixels")

    # Return histogram and x-coordinates of left & right lanes to calculate
    # lane width in pixels
    return histogramL, leftxBase


def slide_window_search_left(binary_warped, histogram):
    # Find the start of left and right lane lines using histogram info
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = int(histogram.shape[0])
    leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # A total of 9 windows will be used
    nwindows = 9
    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    margin = 100
    minpix = 50
    left_lane_inds = []

    #### START - Loop to iterate through windows and search for lane lines #####
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                     (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
    #### END - Loop to iterate through windows and search for lane lines #######

    left_lane_inds = np.concatenate(left_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    # Apply 2nd degree polynomial fit to fit curves
    left_fit = np.polyfit(lefty, leftx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

    ltx = np.trunc(left_fitx)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]

    plt.plot(left_fitx, ploty, color='yellow')
    plt.xlim(0, 480)
    plt.ylim(360, 0)
    # plt.imshow(out_img)

    return left_fit, ltx


def general_search_left(binary_warped, left_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

    ## VISUALIZATION ###########################################################

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    cv.fillPoly(window_img, [np.array(left_line_pts, dtype=int)], (0, 255, 0))
    result = cv.addWeighted(out_img, 1, window_img, 0.3, 0)

    # plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.xlim(0, 480)
    plt.ylim(360, 0)

    ret = {}
    ret['leftx'] = leftx
    ret['left_fitx'] = left_fitx
    ret['ploty'] = ploty

    return ret


def offCenterLeftLane(inpFrame, threshL, minverse):
    hist, leftBase = plotHistogramLeft(threshL)

    left_fit, left_fitx = slide_window_search_left(threshL, hist)

    draw_info = general_search_left(threshL, left_fit)

    mean_x = np.mean(left_fitx, axis=0)



    pass


def get_wheel_position(frame, previous_position, max_wheel_angle):
    """
    Calculates the next wheel angle base on position off axis
    Args:
        frame: 480x360 resolution frame
        previous_position: previous position of the wheels
        max_wheel_angle: maximum angle of the wheels
    Returns:
        wheel angle in degrees, 0 meaning straight, +n meaning right, -n meaning left
    """
    try:
        frame = cv.resize(frame, (480, 360))

        # cv.imshow('frame', frame)

        # Apply perspective warping by calling the "perspectiveWarp()" function
        # Then assign it to the variable called (birdView)
        # Provide this function with:
        # 1- an image to apply perspective warping (frame)
        birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)

        # Apply image processing by calling the "processImage()" function
        # Then assign their respective variables (img, hls, grayscale, thresh, blur, canny)
        # Provide this function with:
        # 1- an already perspective warped image to process (birdView)
        img, hls, grayscale, thresh, blur, canny = processImage(birdView)
        imgL, hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
        imgR, hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)

        cv.imshow("thresh", thresh)

        # Plot and display the histogram by calling the "get_histogram()" function
        # Provide this function with:
        # 1- an image to calculate histogram on (thresh)
        hist, leftBase, rightBase = plotHistogram(thresh)
        # print(rightBase - leftBase)
        # plt.legend([np.sum(threshL) / 255, np.sum(threshR) / 255])
        # plt.plot(hist)
        # plt.show()

        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(thresh, hist)
        # plt.plot(left_fit)
        # plt.show()

        mean_x = np.mean((left_fitx, right_fitx), axis=0)
        meanPts = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

        # draw_info = general_search(thresh, left_fit, right_fit)
        # # plt.show()
        #
        # curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx)
        #
        #
        # # Filling the area of detected lanes with green
        # meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)
        # cv.waitKey(0)

        thresh_height = thresh.shape[0]
        threshL_bottom_half = threshL[thresh_height // 2:, :]
        threshR_bottom_half = threshR[thresh_height // 2:, :]


        # HAPPY CASE BOTH LINES ARE VISIBLE
        if np.sum(threshL_bottom_half) / 255 >= 1000 and np.sum(threshR_bottom_half) / 255 >= 1000:
            print("both lanes")
            deviation, directionDev = offCenter(meanPts, frame)

            max_deviation_meters = 0.35
            # Map deviation to the range [-max_deviation, max_deviation]
            normalized_deviation = deviation / max_deviation_meters

            # Map normalized deviation to wheel angle range [-max_wheel_angle, max_wheel_angle]
            wheel_angle = normalized_deviation * max_wheel_angle * -1

            # Ensures that wheel_angle stays within the range of -max_wheel_angle to max_wheel_angle
            wheel_angle = max(-max_wheel_angle, min(wheel_angle, max_wheel_angle))

            return wheel_angle
        # if LeftBirdView has less than 1 000 piexels means left line doesn't exist
        elif np.sum(threshL_bottom_half) / 255 < 1000:
            print("left lane")
            # wheel_angle = offCenterLeftLane(frame, threshL, minverse)
            return -max_wheel_angle
        elif np.sum(threshR_bottom_half) / 255 < 1000:
            print("right lane")
            return max_wheel_angle
        else:
            print("No lanes")
            return previous_position
    except Exception as error:
        print(error)
        return previous_position


################################################################################
######## START - MAIN FUNCTION #################################################
################################################################################

# Read the input image
image = readVideo()

################################################################################
#### START - LOOP TO PLAY THE INPUT IMAGE ######################################
while True:
    try:

        ret, frame = image.read()
        # Check if frame is read correctly
        if not ret:
            print("Error: Unable to read frame.")
            exit(0)

        # frame = cv.imread("resources/testing_images/right_line.jpg", cv.IMREAD_COLOR)
        frame = cv.resize(frame, (480, 360))

        cv.imshow('original image', frame)

        wheel_position = get_wheel_position(frame, wheel_position, 30)


        # Adding text to our final image
        finalImg = addText(frame,
                           wheel_position)

        # Displaying final image
        cv.imshow("Final", finalImg)
        cv.waitKey(0)

        # Wait for the ENTER key to be pressed to stop playback
        if cv.waitKey(1) == 13:
            break
    except:
        print("ERROR")


# Cleanup
image.release()
cv.destroyAllWindows()

################################################################################
######## END - MAIN FUNCTION ###################################################
################################################################################
