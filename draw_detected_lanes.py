import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import model_from_json

# Load Keras model
json_file = open('full_CNN_model_Logan.json', 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
model.load_weights('full_CNN_model_Logan.h5')



def resize_crop_frame(image, desired_x, crop_ratio):
    
    size = image.shape
    cropped_image = image[100: size[0], 0:size[1]]

    return cropped_image

def image_changes(image):


    edit_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    unsharp_image = cv2.addWeighted(edit_image, 1.5, edit_image, -0.5, 0, edit_image)

    thresh = 240

    edit_image = cv2.threshold(edit_image, thresh, 255, cv2.THRESH_BINARY)[1]

    edit_image = cv2.dilate(edit_image, (7,7))

    #cv2.imshow("thresh", edit_image)

    #edit_image = cv2.Canny(edit_image,200,400)

    #edit_image = cv2.GaussianBlur(edit_image, (3,3), 0 )



    return edit_image

def find_lines(image, draw_image):

    lines = cv2.HoughLinesP(image, 1, np.pi/180, 180, np.array([]) , 5, 5)
    lines_to_return = []

    try:

        for line in lines:
            coords = line[0]
            #print(coords)

            x1=int(coords[0])
            x2=int(coords[2])
            y1=int(coords[1])
            y2=int(coords[3])

            slopex = x2-x1
            slopey = y2-y1

            if slopey != 0:
                slope = float(slopex)/float(slopey)
            else:
                slope = 100

            #print(slopey,slopex,slope)

            x = (x2-x1)**2
            y = (y2-y1)**2
            z = int(x + y)
            length = int(z**(1/2.0))

            #print(slope, length)

            if length > int(10):
                if slope < int(1):
                    if slope > int(-1):
                        lines_to_return += [[x1, y1, x2, y2, slope, length]]
                        cv2.line(draw_image, (x1,y1), (x2,y2), [255,255,0], 3)

        return lines_to_return

    except:
        return [[0,0,0,0,0,0]]


def line_grouping(line_list):

    
    list_negtwo = list()
    list_negone = list()
    list_zero = list()
    list_posone = list()
    list_postwo = list()
    

    for i in range(len(line_listt)):
        line = line_list[i]
        slope = line[4]
        if slope == -2:
            list_negtwo.append(line)

        if slope == -1:
            list_negone.append(line)

        if slope == 0:
            list_zero.append(line)

        if slope == 1:
            list_posone.append(line)

        if slope == 2:
            list_postwo.append(line)


    grouped_list = list()
    grouped_list.append(list_negtwo)
    grouped_list.append(list_negone)
    grouped_list.append(list_zero)
    grouped_list.append(list_posone)
    grouped_list.append(list_postwo)

    return grouped_list


def create_single_line(list_of_lines):

    x1_sum = 0
    x2_sum = 0
    y1_sum = 0
    y2_sum = 0

    length_of_list = len(list_of_lines)

    for i in range(length_of_list):
        x1_sum += list_of_lines[i][0]
        x2_sum += list_of_lines[i][2]
        y1_sum += list_of_lines[i][1]
        y2_sum += list_of_lines[i][3]

    x1_mean = int(x1_sum / length_of_list)
    x2_mean = int(x2_sum / length_of_list)
    y1_mean = int(y1_sum / length_of_list)
    y2_mean = int(y2_sum / length_of_list)

    
    x1 = x1_mean
    x2 = x2_mean
    y1 = y1_mean
    y2 = y2_mean


    single_line = (x1, y1, x2, y2)

    return single_line

    

def draw_single_lines(grouped_list, draw_image):

    for i in range(5):

        if len(grouped_list[i]) > 3:
            single_line = create_single_line(grouped_list[i])
            cv2.line(draw_image, (single_line[0],single_line[1]), (single_line[2],single_line[3]), [0,255,255], 3)


def BirdsEyeView(image):

    image = cv2.resize(image, (1280, 720))
    
    mage = image[0:600, 220:1280]

    image = cv2.resize(image, (1280, 720))

    img_size = (1000, 480)

    src = np.float32([[0, 600], [1280, 600], [0, 400], [1280, 400]])

    dst = np.float32([[390, 500], [610, 500], [-180, -50], [1180, -50]])


    M = cv2.getPerspectiveTransform(src, dst)

    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, M, img_size)

    return warped



def lane_line_detect(image):

    BirdeyeImage = BirdsEyeView(image)

    edit_image = image_changes(BirdeyeImage)

    #line_list = find_lines(edit_image, BirdeyeImage)

    #grouped_list = line_grouping(line_list)

    #draw_single_lines(grouped_list, cropped_image)

    return edit_image



# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255


    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (720, 1280, 3))



    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    

    return result, lane_image

lanes = Lanes()
'''
# Where to save the output video
vid_output = 'proj_reg_vid.mp4'

# Location of the input video
clip1 = VideoFileClip("Test1Curvey.mp4")

vid_clip = clip1.fl_image(road_lines)
vid_clip.write_videofile(vid_output, audio=False)
'''
cap = cv2.VideoCapture("HighwayTest1.mp4")

while True:
    frame, img = cap.read()

    img = imresize(img, (720, 1280, 3))
    result, lane_image = road_lines(img)
    
    
    edit_image = lane_line_detect(img)

    blanks = np.zeros_like(edit_image).astype(np.uint8)
    edit_image = np.dstack((edit_image, edit_image, edit_image))

    lane_image = BirdsEyeView(lane_image)


    #result = cv2.addWeighted(edit_image, 1, lane_image, 1, 0)


    cv2.imshow("original", img)
    cv2.imshow("Roadlines", result)
    cv2.waitKey(10)

