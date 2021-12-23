import cv2
import numpy as np
from keras.models import load_model
import utils
import matplotlib.pyplot as plt

class recognizer:

    model = None
    with_pad = False
    size = 36
    labels = [#'0','1','2','3','4','5','6','7','8','9', # digits are excluded
              'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
              'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    min_h = 75
    min_w = 75
    default_ratio = 0.85

    ##################################################
    # Initialization                                 #
    ##################################################
    def __init__(self, model_name):
        file_path = utils.get_file_path(model_name)
        self.model = load_model(file_path)
        if '_pad' in model_name : self.with_pad = True

    ##################################################
    # Recognize Characters                           #
    ##################################################
    def recognize(self, im):
        chars = self.segment_characters(im)
        result, accuracy = None, None
        if len(chars) > 0:
            result,accuracies = self.recognize_character(chars)
        return result,accuracies
        
    ##################################################
    # Segment  Characters                            #
    ##################################################
    def segment_characters(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        inverse = np.where(gray<200, 1, 0).astype(np.uint8)
        splitted = self.split(inverse)

        chars = []
        if len(splitted) > 0:
            ratios = self.get_ratios(splitted)
            for inv, ratio in zip(splitted, ratios):
                im,filled = utils.refine(inv, self.size, self.with_pad)
                char = {'im' : im, 'ratio' : ratio}
                chars.append(char)
        return chars

    ##################################################
    # Recognize Character                            #
    ##################################################
    def recognize_character(self, chars):
        result = ''
        accuracies = []
        for char in chars:
            res,accuracy = self.predict(char)
            result += res
            accuracies.append(accuracy)
        #accuracy = int(sum(accuracies)/len(accuracies)*100)
        return result,accuracies

    ##################################################
    # Split Characters by Cut                        #
    ##################################################
    def split(self, cropped):
        masked_exp = None # exception for 'i' & 'j'
        last_index = -1
        last_attached = -1
        splitted = []
        is_last = False # 1 character (mask in white) or the last character (inverse mask in black)

        while is_last == False:
            cropped = utils.crop(cropped)
            mask = self.cut(cropped)

            # 'i' & 'j' Check
            masked_cropped = utils.crop(cropped*mask)
            is_small = masked_cropped.shape[0] < self.min_h and masked_cropped.shape[1] < self.min_w

            '''
            # debugging
            plt.imshow(masked_cropped, cmap='gray')
            plt.show()
            '''

            # the top part of 'i' & 'j'
            if masked_exp is None:
                if is_small == True:
                    masked_exp = cropped*mask

                    '''
                    # debugging
                    plt.imshow(masked_exp, cmap='gray')
                    plt.show()
                    '''

                # Other Characters
                else:
                    masked = cropped * mask
                    splitted.append(masked)
                    last_index += 1

            # the bottom part of 'i' & 'j'
            else:
                # the conditions where top part of 'i' & 'j' should not attach to the previously captured character
                if len(splitted) == 0 or last_attached == last_index:
                    is_something = True
                    prev_h_ratio = 0

                # check the previously captured character by shape & size
                else:
                    masked_exp_cropped = utils.crop(masked_exp)
                    pos_y = splitted[last_index].shape[0]-masked_exp.shape[0]
                    is_something = np.max(splitted[last_index][pos_y:pos_y+masked_exp_cropped.shape[0]]).astype(np.bool_)

                    last_splitted = utils.crop(splitted[last_index])
                    prev_h_ratio = np.float32(last_splitted.shape[0]/last_splitted.shape[1])

                #check the current captured character by size
                curr_h_ratio = np.float32(masked_cropped.shape[0]/masked_cropped.shape[1])

                # Attach the top part of 'i' & 'j' to the currently captured character (bottom part of 'i' & 'j')
                if prev_h_ratio < curr_h_ratio or is_something == True:
                    pos_y = masked_exp.shape[0]-cropped.shape[0]
                    pos_x = cropped.shape[1]

                    masked_exp[pos_y:, :pos_x] += cropped * mask

                    '''
                    # debugging
                    print('***')
                    plt.imshow(masked_exp, cmap='gray')
                    plt.show()
                    '''

                    splitted.append(masked_exp)
                    last_index += 1
                    last_attached = last_index
                    masked_exp = None

                # Attach the top part of 'i' & 'j' to the previously captured character (bottom part of 'i' & 'j')
                else:
                    pos_y = splitted[last_index].shape[0]-masked_exp.shape[0]
                    pos_x = splitted[last_index].shape[1]-masked_exp.shape[1]
                    
                    splitted[last_index][pos_y:, pos_x:] = np.where((masked_exp==1)|(splitted[last_index][pos_y:, pos_x:]==1), 1, 0)
                    last_attached = last_index

                    # the top part of 'i' & 'j' is captured twice continuously
                    if is_small == True:
                        masked_exp = cropped*mask
                    else:
                        masked = cropped * mask
                        splitted.append(masked)
                        last_index += 1
                        masked_exp = None

            # 1 Character is only written (mask is all in white)
            is_last = np.min(mask).astype(np.bool_)

            # Multiple characters or parts of character
            if is_last == False:
                mask_inverse = np.where(mask==True, 0, 1).astype(np.bool_)
                cropped *= mask_inverse

                '''
                # debugging
                plt.imshow(cropped, cmap='gray')
                plt.show()
                '''

                # The last character (inverse mask is all in black)
                if np.max(cropped).astype(np.bool_) == 0:
                    if masked_exp is not None:
                        splitted.append(masked_exp)
                        last_index += 1
                        masked_exp = None
                    is_last = True
                else:
                    masked_cropped = utils.crop(cropped)
                    is_small = masked_cropped.shape[0] < self.min_h and masked_cropped.shape[1] < self.min_w
                    
                    # Attach the remaining top part of 'i' & 'j' to the previously captured character
                    if is_small == True:
                        pos_y = splitted[last_index].shape[0]-cropped.shape[0]
                        pos_x = splitted[last_index].shape[1]-cropped.shape[1]
                        splitted[last_index][pos_y:, pos_x:] = np.where((cropped==1)|(splitted[last_index][pos_y:, pos_x:]==1), 1, 0)
                        is_last = True
        return splitted

    ##################################################
    # Split Characters by gap (attempted)            #
    ##################################################
    def split_(self, cropped):
        height,width = cropped.shape
        gap_width = 10

        splitted = []
        x = 0
        x_ = 0
        while x < width-gap_width+1:
            if np.sum(cropped[:, x:x+gap_width]) == 0:
                split = cropped[:, x_:x]
                if np.sum(split) > 0:
                    splitted.append(split)
                x_ = x+gap_width
                x = x_
            else:
                x += 1

        split = cropped[:, x_:]
        if np.sum(split) > 0:
            splitted.append(split)
        return splitted

    ##################################################
    # Cut (Shortest Path)                            #
    ##################################################
    def cut(self, cropped):
        h,w = cropped.shape
        pad = np.ones((h, 1), np.uint8)*1e10
        sums = np.hstack((pad, cropped, pad))

        # Forward for the sum of each path
        for i in range(1, h):
            for j in range(1, w+1):
                choices = sums[i-1, j-1:j+2]
                index_min = np.argmin(choices, axis=0) + (j-1)
                sums[i, j] += sums[i-1, index_min]

        # The minimum sum at the endpoint
        j = np.argmin(sums[h-1, :], axis=0)
        mask = np.zeros((h, w+2), np.bool_)
        mask[h-1:, j] = 1

        # Backward for determining the shortest path
        for i in range(h-2, -1, -1):
            choices = sums[i, j-1:j+2]
            j = np.argmin(choices, axis=0) + (j-1)
            
            # 1 Character or overrapping characters
            if cropped[i, j-1] > 0: return np.ones((h, w), np.bool_)

            mask[i, :j] = 1
        mask = mask[:, 1:w+1]
        return mask

    ##################################################
    # Get Ratios of Images                           #
    ##################################################
    def get_ratios(self, splitted):
        ratios = []
        for splt in splitted:
            ratios.append(np.float32(splt.shape[0]))
        ratios /= max(ratios)
        return ratios
    
    ##################################################
    # Predict Character                              #
    ##################################################
    def predict(self, char):
        im = char['im'].astype(np.float32)

        # Reshape for prediction
        im = im.reshape(1,self.size,self.size,1)
        #im /= 255.0
        
        # Predict
        res = self.model.predict([im])[0]

        '''
        # debugging
        print(res)
        '''

        label = self.labels[np.argmax(res)] # label by the index of the highest accuracy
        accuracy = max(res)
        return label,accuracy
