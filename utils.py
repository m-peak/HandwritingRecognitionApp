import os
import sys
import cv2
import numpy as np
from PIL import ImageGrab, Image
import matplotlib.pyplot as plt

##################################################
# Get Application Path                           #
# (& absolute path to resources for PyInstaller) #
##################################################
def get_file_path(file_name):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, file_name)
    return file_path

##################################################
# Get OS                                         #
##################################################
def get_os():
    oss = {'darwin' : 'mac', 'win32' : 'win'}
    return oss[sys.platform]

##################################################
# Capture Handwriting                            #
##################################################
def capture_handwriting(app_title, canvas_width, canvas_height, winfo_id):
    window_size = {'X' : 0, 'Y' : 0}
    canvas_y = 0
    os_name = get_os()
    if os_name == 'mac':
        from Quartz import kCGWindowListOptionOnScreenOnly, kCGNullWindowID, CGWindowListCopyWindowInfo
        
        '''
        # debugging
        from Quartz import CGDisplayBounds, CGMainDisplayID
        main_monitor = CGDisplayBounds(CGMainDisplayID())
        print(main_monitor.size.width, main_monitor.size.height)
        '''
        
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        for window in window_list:
            if 'kCGWindowName' in window:
                if window['kCGWindowName'] == app_title:
                    window_size['X'] = window['kCGWindowBounds']['X']
                    window_size['Y'] = window['kCGWindowBounds']['Y']
        canvas_y = 25
    elif os_name == 'win':
        import win32gui
        window_rect = win32gui.GetWindowRect(winfo_id)
        window_size['X'] = window_rect[0]
        window_size['Y'] = window_rect[1]
        canvas_width -= 4

    canvas_x = window_size['X']
    canvas_y += window_size['Y']
    image = ImageGrab.grab(bbox=(canvas_x, canvas_y, canvas_x+canvas_width, canvas_y+canvas_height))
    im = np.array(image).astype(np.uint8)
    return im

##################################################
# Resize Inversed Image                          #
##################################################
def resize(cropped, size, with_pad=False):
    adjusted = cropped
    y,x = cropped.shape
    diff = abs(y-x)

    # Adjust image by padding only if the difference is twice
    if with_pad == True:
        if diff > y or diff > x:

            '''
            # debugging
            print(y, x)
            '''
            
            diff_half = int(diff/2)
            s = y if y > x else x
            adjusted = np.zeros((s,s), np.uint8)

            if y > x:
                adjusted[:, diff_half:diff_half+x] = cropped
            else:
                adjusted[diff_half:diff_half+y, :] = cropped
    
    resized = cv2.resize(adjusted, (size,size))
    return resized

##################################################
# Crop Inversed Image                            #
##################################################
def crop(inverse):
    # remove little noise vertically
    mask_vertical = np.where(np.sum(inverse, axis=0) == 1)
    inverse[:, mask_vertical] = 0
    
    # remove little noise horizontally
    h = inverse.shape[0]
    if np.sum(inverse[h-2]) == 0: inverse[h-1] = 0

    indices = np.nonzero(inverse)
    y_max = inverse.shape[0] if len(indices[0])==0 else max(indices[0])
    x_max = inverse.shape[1] if len(indices[1])==0 else max(indices[1])
    y_min = 0 if len(indices[0])==0 else min(indices[0])
    x_min = 0 if len(indices[1])==0 else min(indices[1])
    cropped = inverse[y_min:y_max+1, x_min:x_max+1]
    return cropped

##################################################
# Set Start & End Indices                        #
##################################################
def set_indices(vector, size):
    ones = []
    for i in range(1, size):
        if vector[i] == 1:
            if i == 1 or i == size-1:
                ones.append(i)
            else:
                if vector[i-1] == 0:
                    ones.append(i)
        else:
            if i > 1 and vector[i-1] == 1:
                ones.append(i-1)
    return ones

##################################################
# Fill Inversed Image                            #
##################################################
def fill(filled, size):

    # Check start and end locations of edges in X-axis (each row)
    for y in range(1, size):
        ones = set_indices(filled[y], size)

        # Check start and end locations of ones[] in Y-axis (each column in each row)
        cnt = len(ones)
        # White pixel has to appear at least twice (at least 3 locations = start, end and start points)
        if cnt > 2:
            i = 1
            while i < cnt-1:
                w1 = ones[i]+1 # start of blank pixel
                w2 = ones[i+1]-1 # end of blank pixel
                ws = [w1, w2]
                ones2 = []
                for w in ws:
                    ones3 = set_indices(filled[:, w], size)
                    ones2.append(ones3)

                cnt2_0 = len(ones2[0]) # start and end locations of pixel value 1 for w1 (the start of white pixel in Y-axis)
                cnt2_1 = len(ones2[1]) # start and end locations of pixel value 1 for w2 (the end of white pixel in Y-axis)

                # Fill white pixels in an Enclosed area
                # White pixel has to appear at least twice (at least 3 locations = start, end and start points)
                # The same pixel locations in X-axis in the previous row have the pixel value of 1
                if cnt2_0 > 2 and cnt2_1 > 2 and np.sum(filled[y-1, w1:w2]) == w2-w1:
                    j = 1
                    while j < cnt2_0-1 and j < cnt2_1-1:
                        
                        # Fill in the Maximum area (the minimum and the maximum locations in Y-axis in the block of the blank area)
                        h = min(ones2[0][j], ones2[1][j])
                        h_ = max(ones2[0][j+1], ones2[1][j+1])
                        if h < y and y < h_:
                            filled[y, w1:w2+1] = 1
                        j += 2
                i += 2
    return filled     

##################################################
# Refine Inversed Image                          #
##################################################
def refine(inverse, size, with_pad=False, with_fill=False):
    cropped = crop(inverse)
    resized = resize(cropped, size, with_pad)
    
    filled = None
    if with_fill == True:
        filled = fill(resized.copy(), size).astype(np.float32)
        mask = np.where(filled-resized == 1, True, False).astype(np.bool_) # mask for gray area
        filled[mask] = 0.5

        '''
        # debugging
        plt.imshow(resized, cmap='gray')
        plt.show()
        plt.imshow(filled, cmap='gray')
        plt.show()
        '''

    return resized,filled
