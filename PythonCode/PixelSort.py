import cv2
import os
import sys
import numpy as np
import io
import base64
from PIL import Image
from shiny import App, reactive, render, ui, render_image
from shiny.types import FileInfo

app_ui = ui.page_sidebar(
    # Parameter Sidebar
    ui.sidebar(
        ui.input_file('dataupload', 'Upload Image',
                      multiple=False, accept=['.jpeg','.jpg','.png']
        ),
        
        ui.input_selectize("sortcolor", "Select Sort Channel:",  
        {"red": "Red", "green": "Green", "blue": "Blue",
         "hue": "Hue", "saturation": "Saturation", "lightness": "Lightness"} 
        ),
        ui.input_selectize("sortdirec", "Select Sort Direction:",  
        {"up": "Bottom-to-Top", "down": "Top-to-Bottom",
         "right": "Left-to-Right", "left": "Right-to-Left"}
        ),
        ui.input_slider("threshold", "Threshold Range", min=0, max=255, value=[0, 255]),
        ui.input_switch("applysort", "Pixel Sort!", False),
        title="Pixel Sort Parameters",
    ),
    
    # Upload and Display
    ui.card(
        ui.output_ui("display_img"),
    ),
    fillable=True,
)


def server(input, output, session):
    def image_to_base64(image):
        _, buffer = cv2.imencode('.png', image)
        img_bytes = buffer.tobytes()
        return base64.b64encode(img_bytes).decode()

    def grab_cv2img_col(cv2_img, colidx):
        output = [0] * len(cv2_img)
        for rowidx in range(len(output)):
            output[rowidx] = cv2_img[rowidx][colidx]
        return(output)

    def replace_cv2img_col(cv2_img, colidx, newcol):
        for rowidx in range(len(newcol)):
            cv2_img[rowidx][colidx] = newcol[rowidx]
        return(cv2_img)

    def sort_img_column(img_col, color_value_idx=None, lower_thresh=None, upper_thresh=None):
        # Get sortable values (no threshold yet)
        if color_value_idx is None or type(img_col[0]) == np.uint8: # for black/white images, just use the values there
            simple_col = img_col
        else:
            simple_col = [0] * len(img_col) # pull value at desired index for rgb of hsv images
            for idx in range(len(img_col)):
                simple_col[idx] = img_col[idx][color_value_idx]

        # Set lower threshold if None
        if lower_thresh is None:
            lower_thresh = min(simple_col)
        
        # Set upper threshold if None
        if upper_thresh is None:
            upper_thresh = max(simple_col)

        # Get intervals for sorting (control variables)
        interval_values, interval_start_idxs = [], [] # store the simple_col values and the index the intervals start at
        mid_interval_flag = False # communicates if we are in an interval
        
        # Get intervals for sorting (loop)
        for idx in range(len(simple_col)):
            thresh_bool = simple_col[idx] > lower_thresh and simple_col[idx] < upper_thresh # value is within thresholds
            
            if thresh_bool and mid_interval_flag is False: # Start a new interval
                interval_start_idxs.append(idx)
                interval_values.append([simple_col[idx]])
                mid_interval_flag = True
                
            elif thresh_bool and mid_interval_flag is True: # Add to current interval
                interval_values[-1].append(simple_col[idx])

            elif (not thresh_bool) and mid_interval_flag is True: # End current interval
                mid_interval_flag = False

        # Assemble sorted column control variables
        newcol = [0] * len(img_col) # ultimate output
        next_interval = 0 # idx of interval in interval_values and interal_start_idxs to use at a given time
        merge_idx = 0 # idx for newcol placement

        # Assemble sorted column loop
        while merge_idx < len(simple_col):
            if next_interval < len(interval_values) and merge_idx == interval_start_idxs[next_interval]: # if at an interval
                sort_key = np.argsort(interval_values[next_interval]) + merge_idx # get sorted indexes 
                
                for idx in range(len(interval_values[next_interval])): # add each img_col value(s) per sorted index
                    newcol[merge_idx + idx] = img_col[sort_key[idx]]
                            
                merge_idx += len(interval_values[next_interval]) # move merge_idx past the interval
                next_interval += 1
            else:
                newcol[merge_idx] = img_col[merge_idx] # add non-interval values as is
                merge_idx += 1 # update idx variable
        
        return(np.array(newcol))
         
    def sort_image(cv2_img):
        # Select color index and convert to hls if needed
        if input.sortcolor() == 'red':
            color_idx = 2
        elif input.sortcolor() == 'green':
            color_idx = 1
        elif input.sortcolor() == 'blue':
            color_idx = 0
        elif input.sortcolor() == 'hue':
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HLS)
            color_idx = 0
        elif input.sortcolor() == 'saturation':
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HLS)
            color_idx = 2
        else:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HLS)
            color_idx = 1
            
        # Rotate image to allow different sort directions
        if input.sortdirec() == "left":
            cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_CLOCKWISE)
        elif input.sortdirec() == "up":
            cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_180)
        elif input.sortdirec() == "right":
            cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for colidx in range(len(cv2_img[0])):
            grabbed_col = grab_cv2img_col(cv2_img, colidx)
            grabbed_col = sort_img_column(grabbed_col, color_idx, 
                            input.threshold()[0], input.threshold()[1])
            output = replace_cv2img_col(cv2_img, colidx, grabbed_col)
            
        # Unrotate image
        if input.sortdirec() == "left":
            output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif input.sortdirec() == "up":
            output = cv2.rotate(output, cv2.ROTATE_180)
        elif input.sortdirec() == "right":
            output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
            
        # Unconvert colors if needed
        if input.sortcolor() in ('hue', 'saturation', 'lightness'):
            output = cv2.cvtColor(output, cv2.COLOR_HLS2BGR)
        
        return(output)
        
    @reactive.calc
    def rawdata():
        file: list[FileInfo] | None = input.dataupload()
        if file is None:
            return cv2.imread('./ExampleImage.jpg')
        datapath = file[0]["datapath"]
        return cv2.imread(datapath)
    
    @render.ui 
    def display_img():
        if input.applysort():
            img = sort_image(rawdata())
        else:
            img = rawdata()
        
        # Convert the image to base64
        img = image_to_base64(img)
        return ui.img(src=f"data:image/png;base64,{img}", alt="Display Image/Frame")


app = App(app_ui, server)
