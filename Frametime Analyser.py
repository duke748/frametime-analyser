# -*- coding: utf-8 -*-
import sys
import os
import time
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from libraries import *

pause_frame = False 
# pause_frame = True #wait for keypress before playing next frame

# Parse command line arguments
target_fps = 60  # Default to 60fps
show_delta_plot = False  # Default to not showing delta plot
watermark_file = None  # Default to no watermark
title_text = None  # Default to no title
detect_stutters = False  # Default to not detecting stutters

if len(sys.argv) > 1:
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        sys.exit(1)
    
    # Check for fps argument and optional flags
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-delta-plot":
            show_delta_plot = True
            i += 1
        elif arg == "-detect-stutters":
            detect_stutters = True
            i += 1
        elif arg == "-watermark":
            if i + 1 < len(sys.argv):
                watermark_file = sys.argv[i + 1]
                if not os.path.exists(watermark_file):
                    print(f"Error: Watermark file '{watermark_file}' not found!")
                    sys.exit(1)
                i += 2
            else:
                print("Error: -watermark flag requires a file path")
                print("Usage: python 'Frametime Analyser.py' <path_to_video_file> [60|120] [-delta-plot] [-watermark <image_file>] [-title <text>] [-detect-stutters]")
                sys.exit(1)
        elif arg == "-title":
            if i + 1 < len(sys.argv):
                title_text = sys.argv[i + 1]
                i += 2
            else:
                print("Error: -title flag requires a text string")
                print("Usage: python 'Frametime Analyser.py' <path_to_video_file> [60|120] [-delta-plot] [-watermark <image_file>] [-title <text>] [-detect-stutters]")
                sys.exit(1)
        else:
            try:
                target_fps = int(arg)
                if target_fps not in [60, 120]:
                    print("Error: FPS must be either 60 or 120")
                    print("Usage: python 'Frametime Analyser.py' <path_to_video_file> [60|120] [-delta-plot] [-watermark <image_file>] [-title <text>] [-detect-stutters]")
                    sys.exit(1)
                i += 1
            except ValueError:
                print(f"Error: Invalid argument '{arg}'")
                print("Usage: python 'Frametime Analyser.py' <path_to_video_file> [60|120] [-delta-plot] [-watermark <image_file>] [-title <text>] [-detect-stutters]")
                sys.exit(1)
else:
    # Default file paths (commented out examples)
    # file_path="F:/ReLive/rdr2 h264.m4v" #transcoded HEVC to h264
    # file_path="F:/ReLive/2020.10.17-09.29.mp4" #second h264
    # file_path="F:/ReLive/2020.09.24-21.36.mp4" #ACO capture
    # file_path="E:/Downloads/The Last of Us 2 - What 60fps Gameplay Looks Like.mp4"
    file_path="F:/ReLive/2020.10.14-21.51.mp4" #forza
    
    if not os.path.exists(file_path):
        print("Error: No video file specified and default file not found!")
        print("Usage: python 'Frametime Analyser.py' <path_to_video_file> [60|120] [-delta-plot] [-watermark <image_file>] [-title <text>] [-detect-stutters]")
        sys.exit(1)

print(f"Loading video: {file_path}")
print(f"Target FPS: {target_fps}")
print(f"Delta Plot: {'Enabled' if show_delta_plot else 'Disabled'}")
print(f"Watermark: {watermark_file if watermark_file else 'Disabled'}")
print(f"Title: {title_text if title_text else 'Disabled'}")
print(f"Stutter Detection: {'Enabled' if detect_stutters else 'Disabled'}")

# Calculate frame time in milliseconds
target_frametime = 1000.0 / target_fps  # 16.67ms for 60fps, 8.33ms for 120fps

# Set up video writer for output
output_path = os.path.splitext(file_path)[0] + "_analyzed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None  # Will be initialized after we know the frame dimensions

fvs = FileVideoStream(file_path).start()
time.sleep(3)
# start the FPS timer
# fps = FPS().start()



count = 0
frametime = 0  # Counts frames since last unique frame
frametime_text = "None"


frametime_samples = 90  # Increased to fill the 270px box at 3px spacing
frametime_graph = [0]*frametime_samples #list of previous frametimes

# Frame-to-Frame Delta tracking for micro-stutter detection
frametime_delta_graph = [0]*frametime_samples  # Stores |Δ frametime| between consecutive frames

# Stutter detection tracking
stutter_events = []  # List of (frame_number, frametime_ms, stutter_type) tuples
stutter_threshold_multiplier = 2.5  # Frametime must be 2.5x target to count as stutter
severe_stutter_threshold_multiplier = 4.0  # Severe stutter (shader compilation, etc.)

# Frametime histogram data - buckets for different frametime ranges
# Will store last 30 minutes of data for histogram
# At 60fps: 60 * 60 * 30 = 108,000 unique frames
# At 120fps: 120 * 60 * 30 = 216,000 unique frames
frametime_history = []
histogram_sample_size = 216000  # 30 minutes at 120fps (covers both modes)

# FPS list always represents 1 second of data (60 samples for 60fps base rate)
# This ensures FPS calculation is consistent regardless of target_fps
fps_list = [0]*60 # 0 means repeated frame, 1 means unique frame. summation of fps_list gives the current fps
fps = 0
fps_display = ""  # FPS value shown on screen (updates slower)
fps_display_timer = 0  # Timer for FPS display update
fps_display_update_interval = 30  # Update FPS display every 30 frames (~0.5 seconds at 60fps)
fps_graph_samples = 580
fps_graph = [0]*fps_graph_samples


perf_list = [0] #list of miliseconds taken for each segment of code
timestamp_before_imshow = None #timestamp to be taken before cv2.imshow
moving_avg_brightness = 0	#exponentially decaying moving avg of previous frames' brightness

# Load watermark image if specified
watermark = None
if watermark_file:
    watermark = cv2.imread(watermark_file, cv2.IMREAD_UNCHANGED)
    if watermark is not None:
        # Resize to 32x32
        watermark = cv2.resize(watermark, (32, 32))
        print(f"Watermark loaded: {watermark_file}")
    else:
        print(f"Warning: Could not load watermark from {watermark_file}")

while fvs.more():  
    if count == 0:
        frame = fvs.read()
        prev_frame = frame
        height = len(frame)
        width = len(frame[0])
        timestamp_before_imshow = cv2.getTickCount()
        
        # Initialize video writer with frame dimensions
        resize_height, resize_width = frame.shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, float(target_fps), (resize_width, resize_height))
        print(f"Output will be saved to: {output_path}")
    perf_list.append((cv2.getTickCount() - timestamp_before_imshow )/ cv2.getTickFrequency()*1000-sum(perf_list))

    # text_color = (0,0,0)
    text_color = (255,255,255)
    # graph_color = (128,0,0)
    graph_color = (93, 232, 130)
    shadow_color = (0,0,0)

    src_frame = fvs.read()
    # frame = src_frame
    frame = src_frame.copy()
    
    # Increment frametime at start of each frame processing
    frametime = frametime + 1
    
    perf_list.append((cv2.getTickCount() - timestamp_before_imshow)/ cv2.getTickFrequency()*1000-sum(perf_list))

    # cv2.imshow('frame',prev_frame)

    # #downscale before calculating difference
    scale = 0.1
    cropped_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    cropped_prev_frame = cv2.resize(prev_frame, (0,0), fx=scale, fy=scale) 
    

    #convert to grayscale
    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    cropped_prev_frame = cv2.cvtColor(cropped_prev_frame, cv2.COLOR_BGR2GRAY)


    #calculate frame difference
    frame_diff = cv2.absdiff(cropped_frame, cropped_prev_frame)
    average = frame_diff.mean(axis=0).mean(axis=0)
    if count == 1:
        moving_avg_brightness = average
    perf_list.append((cv2.getTickCount() - timestamp_before_imshow)/ cv2.getTickFrequency()*1000-sum(perf_list))
        


    threshold = 0.25*moving_avg_brightness #threshold below which a frame is considered "identical" or dropped
    if average < threshold:
        fps_list.pop(0)
        fps_list.append(0)
        fps = sum(fps_list)

        fps_graph.pop(0)
        fps_graph.append(fps)
    else:
        result = ""
        # Record frametime (already incremented at end of previous loop)
        # Convert frame count to actual milliseconds
        frametime_text = frametime * target_frametime
        
        # Calculate frame-to-frame delta for micro-stutter detection
        current_ft = frametime * target_frametime
        previous_ft = frametime_graph[-1] if frametime_graph[-1] > 0 else current_ft
        frametime_delta = abs(current_ft - previous_ft)
        
        # Stutter detection: Check if current frametime exceeds threshold
        if detect_stutters:
            stutter_threshold = target_frametime * stutter_threshold_multiplier
            severe_stutter_threshold = target_frametime * severe_stutter_threshold_multiplier
            
            if current_ft >= severe_stutter_threshold:
                stutter_events.append((count, current_ft, "severe"))
            elif current_ft >= stutter_threshold:
                stutter_events.append((count, current_ft, "moderate"))
        
        frametime_graph.pop(0)
        frametime_graph.append(current_ft)
        
        frametime_delta_graph.pop(0)
        frametime_delta_graph.append(frametime_delta)
        
        # Add to histogram history
        frametime_history.append(current_ft)
        if len(frametime_history) > histogram_sample_size:
            frametime_history.pop(0)
        
        fps_list.pop(0)
        fps_list.append(1)
        fps = sum(fps_list)

        fps_graph.pop(0)
        fps_graph.append(fps)
        frametime = 0	#frametime will be incremented at end of loop


    # Don't display FPS until we have at least 60 frames of data
    if(count)<60:
        fps_display = ""
    else:
        # Update FPS display every N frames for smoother visual
        fps_display_timer += 1
        if fps_display_timer >= fps_display_update_interval:
            fps_display = str(fps)
            fps_display_timer = 0
    
    # print (average, moving_avg_brightness, result)
    moving_avg_brightness = (moving_avg_brightness + average/3)*3/4



    # text_to_write = '{:4.2f}'.format(average)+" "+str(frametime_text)+" "+result
    # texted_image =cv2.putText(frame, text=text_to_write, org=(200,200),fontFace=3, fontScale=3, color=(0,0,255), thickness=5)
    

    perf_list.append((cv2.getTickCount() - timestamp_before_imshow)/ cv2.getTickFrequency()*1000-sum(perf_list))


    text_to_write = fps_display
    
    # Draw FPS text with black border
    # texted_image =cv2.putText(frame, text=text_to_write, org=(1140,150),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.2, color=(0,0,0), thickness=5)
    # Draw black border for FPS
    for dx in [-2, 0, 2]:
        for dy in [-2, 0, 2]:
            if dx != 0 or dy != 0:
                cv2.putText(frame, text=text_to_write, org=(1140+dx,150+dy), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(0,0,0), thickness=5, lineType=cv2.LINE_AA)
    # Draw main FPS text
    texted_image = cv2.putText(frame, text=text_to_write, org=(1140,150), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=text_color, thickness=5, lineType=cv2.LINE_AA)
    # texted_image =cv2.putText(frame, text="hello", org=(750,150),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255,255,255), thickness=5)

    # resize = ResizeWithAspectRatio(texted_image, width=1280) #slow function, about 2.5ms
    resize = texted_image
    perf_list.append((cv2.getTickCount() - timestamp_before_imshow)/ cv2.getTickFrequency()*1000-sum(perf_list))


    # Helper function to draw text with black border (defined once, used throughout)
    def draw_text_with_border(img, text, org, fontFace, fontScale, color, thickness, lineType):
        # Draw black border
        border_color = (0, 0, 0)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cv2.putText(img, text=text, org=(org[0]+dx, org[1]+dy),
                               fontFace=fontFace, fontScale=fontScale, 
                               color=border_color, thickness=thickness, lineType=lineType)
        # Draw main text
        cv2.putText(img, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                   color=color, thickness=thickness, lineType=lineType)

    #drawing frametime graph
    pt_width = 3
    pt_spacing = 3 #distance between center of 2 points
    pt_gap = pt_spacing - pt_width
    
    plt_origin_x = 60
    plt_origin_y = 420  # Moved down to align with histogram
    
    # Make graph dimensions match histogram better
    frametime_graph_width = 270  # Match histogram width
    frametime_graph_height = 100  # Match histogram height
    
    # Define graph bounds
    graph_top = plt_origin_y - frametime_graph_height
    graph_bottom = plt_origin_y
    max_frametime_display = target_frametime * 2.5  # Display up to 2.5x target frametime (e.g., 41.7ms for 60fps, 20.8ms for 120fps)

    # Recalculate samples to fit new width
    frametime_samples_display = frametime_graph_width // pt_spacing
    
    # print frametime_graph
    for key, value in enumerate(frametime_graph):
        if key >= frametime_samples_display:
            break

        prev_ft = frametime_graph[key-1] if key > 0 else 0
        if key == 0:
            continue
        if prev_ft == 0:
            continue
        
        # Clip values to stay within the graph bounds
        value_clipped = min(value, max_frametime_display)
        prev_ft_clipped = min(prev_ft, max_frametime_display)
        
        # Calculate y positions and clip to box bounds
        # Scale so that max_frametime_display reaches the top of the graph
        scale_factor = frametime_graph_height / max_frametime_display
        y_value = max(graph_top, plt_origin_y - value_clipped * scale_factor)
        y_prev = max(graph_top, plt_origin_y - prev_ft_clipped * scale_factor)
        
        cv2.line(resize,(
            plt_origin_x + key*pt_spacing, 
            int(y_value)
            ),(
            plt_origin_x + key*pt_spacing + pt_width, 
            int(y_value)
            ),graph_color,2)
        if key !=0:
            cv2.line(resize,(
                plt_origin_x + key*pt_spacing - pt_gap, 
                int(y_prev)
                ),(
                plt_origin_x + key*pt_spacing, 
                int(y_value)
                ),graph_color,2) #vertical
    cv2.rectangle(resize,(plt_origin_x,graph_top),(plt_origin_x+frametime_graph_width,plt_origin_y),text_color,2)
    
    # Draw stutter markers on frametime graph if detection enabled
    if detect_stutters:
        # Get the current viewing window of frames
        frames_in_view = frametime_samples_display
        current_frame_start = count - frames_in_view
        
        for stutter_frame, stutter_ft, stutter_type in stutter_events:
            # Check if this stutter is within the current viewing window
            if current_frame_start <= stutter_frame <= count:
                # Calculate position on the graph
                frame_offset = stutter_frame - current_frame_start
                x_pos = plt_origin_x + frame_offset * pt_spacing
                
                # Draw a vertical line marker
                if stutter_type == "severe":
                    marker_color = (0, 0, 255)  # Red for severe stutter
                    marker_thickness = 3
                else:
                    marker_color = (0, 140, 255)  # Orange for moderate stutter
                    marker_thickness = 2
                
                cv2.line(resize, (x_pos, graph_top), (x_pos, graph_bottom), marker_color, marker_thickness)
    
    # Display frametime markers based on target FPS
    draw_text_with_border(resize, f"{target_frametime:.1f}", (plt_origin_x+frametime_graph_width+10,plt_origin_y-5), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    draw_text_with_border(resize, f"{target_frametime*2:.1f}", (plt_origin_x+frametime_graph_width+10,graph_top+5), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    draw_text_with_border(resize, "FRAMETIME (MS)", (plt_origin_x,graph_top-15), cv2.FONT_HERSHEY_DUPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # Draw Frame-to-Frame Delta Plot (Micro-stutter detection) - Only if enabled
    if show_delta_plot:
        delta_origin_x = plt_origin_x + frametime_graph_width + 80  # Position to the right of frametime graph
        delta_origin_y = plt_origin_y  # Same Y as frametime graph
        delta_graph_width = 270  # Same width as frametime graph
        delta_graph_height = 100  # Same height
        
        delta_graph_top = delta_origin_y - delta_graph_height
        delta_graph_bottom = delta_origin_y
        # Max delta to display - larger deltas indicate micro-stutter
        max_delta_display = target_frametime * 1.5  # Display deltas up to 1.5x target frametime
        
        delta_samples_display = delta_graph_width // pt_spacing
        
        # Color for delta graph - use orange/red to highlight stutter
        delta_color = (0, 165, 255)  # Orange
        
        for key, value in enumerate(frametime_delta_graph):
            if key >= delta_samples_display:
                break

            prev_delta = frametime_delta_graph[key-1] if key > 0 else 0
            if key == 0:
                continue
            if prev_delta == 0 and value == 0:
                continue
            
            # Clip values to stay within the graph bounds
            value_clipped = min(value, max_delta_display)
            prev_delta_clipped = min(prev_delta, max_delta_display)
            
            # Calculate y positions
            scale_factor = delta_graph_height / max_delta_display
            y_value = max(delta_graph_top, delta_origin_y - value_clipped * scale_factor)
            y_prev = max(delta_graph_top, delta_origin_y - prev_delta_clipped * scale_factor)
            
            # Use color coding: green for small deltas, orange for medium, red for large
            if value < target_frametime * 0.5:
                line_color = (93, 232, 130)  # Green - smooth
            elif value < target_frametime:
                line_color = (0, 255, 255)  # Yellow - noticeable
            else:
                line_color = (0, 0, 255)  # Red - micro-stutter
            
            cv2.line(resize,(
                delta_origin_x + key*pt_spacing, 
                int(y_value)
                ),(
                delta_origin_x + key*pt_spacing + pt_width, 
                int(y_value)
                ),line_color,2)
            if key != 0:
                cv2.line(resize,(
                    delta_origin_x + key*pt_spacing - pt_gap, 
                    int(y_prev)
                    ),(
                    delta_origin_x + key*pt_spacing, 
                    int(y_value)
                    ),line_color,2)
        
        cv2.rectangle(resize,(delta_origin_x,delta_graph_top),(delta_origin_x+delta_graph_width,delta_origin_y),text_color,2)
        
        # Display delta markers
        draw_text_with_border(resize, "0", (delta_origin_x+delta_graph_width+10,delta_origin_y-5), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
        draw_text_with_border(resize, f"{max_delta_display:.1f}", (delta_origin_x+delta_graph_width+10,delta_graph_top+5), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
        draw_text_with_border(resize, "FRAME DELTA (MS)", (delta_origin_x,delta_graph_top-15), cv2.FONT_HERSHEY_DUPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # Draw Cumulative Distribution Function (CDF) - Smoothness Graph
    # Only draw if we have enough frametime data
    if len(frametime_history) > 10:
        # Position: to the right of delta plot if enabled, otherwise to the right of frametime graph
        if show_delta_plot:
            cdf_origin_x = plt_origin_x + frametime_graph_width + 80 + delta_graph_width + 80
        else:
            cdf_origin_x = plt_origin_x + frametime_graph_width + 80  # Same position as delta plot would be
        
        cdf_origin_y = plt_origin_y  # Same Y as other graphs
        cdf_graph_width = 270
        cdf_graph_height = 100
        
        cdf_graph_top = cdf_origin_y - cdf_graph_height
        cdf_graph_bottom = cdf_origin_y
        
        # Sort frametime history for CDF calculation
        sorted_frametimes = sorted(frametime_history)
        total_frames = len(sorted_frametimes)
        
        # Determine X-axis range (frametime in ms)
        max_frametime_cdf = target_frametime * 3  # Show up to 3x target frametime
        min_frametime_cdf = 0
        
        # Calculate CDF points - sample at regular intervals across the frametime range
        cdf_samples = 100  # Number of points to plot
        cdf_points = []
        
        for i in range(cdf_samples + 1):
            # Calculate frametime threshold for this point
            ft_threshold = min_frametime_cdf + (max_frametime_cdf - min_frametime_cdf) * (i / cdf_samples)
            
            # Count how many frames are <= this threshold
            count = sum(1 for ft in sorted_frametimes if ft <= ft_threshold)
            percentage = (count / total_frames) * 100 if total_frames > 0 else 0
            
            cdf_points.append((ft_threshold, percentage))
        
        # Draw CDF curve
        for i in range(1, len(cdf_points)):
            ft_prev, pct_prev = cdf_points[i-1]
            ft_curr, pct_curr = cdf_points[i]
            
            # Convert to pixel coordinates
            # X: frametime (0 to max_frametime_cdf) maps to graph width
            # Y: percentage (0 to 100) maps to graph height
            x_prev = cdf_origin_x + int((ft_prev / max_frametime_cdf) * cdf_graph_width)
            x_curr = cdf_origin_x + int((ft_curr / max_frametime_cdf) * cdf_graph_width)
            
            y_prev = cdf_origin_y - int((pct_prev / 100) * cdf_graph_height)
            y_curr = cdf_origin_y - int((pct_curr / 100) * cdf_graph_height)
            
            # Clamp to bounds
            y_prev = max(cdf_graph_top, min(cdf_graph_bottom, y_prev))
            y_curr = max(cdf_graph_top, min(cdf_graph_bottom, y_curr))
            
            # Color based on steepness - steeper = more consistent (better)
            # A steep curve means most frames are fast
            if pct_curr >= 90:
                line_color = (93, 232, 130)  # Green - excellent consistency
            elif pct_curr >= 70:
                line_color = (0, 255, 255)  # Yellow - good
            else:
                line_color = (0, 165, 255)  # Orange - building up
            
            cv2.line(resize, (x_prev, y_prev), (x_curr, y_curr), line_color, 2)
        
        # Draw reference lines at key percentiles
        # 50th percentile (median) line
        median_y = cdf_origin_y - int(0.5 * cdf_graph_height)
        cv2.line(resize, (cdf_origin_x, median_y), (cdf_origin_x + cdf_graph_width, median_y), (128, 128, 128), 1)
        
        # 95th percentile line
        p95_y = cdf_origin_y - int(0.95 * cdf_graph_height)
        cv2.line(resize, (cdf_origin_x, p95_y), (cdf_origin_x + cdf_graph_width, p95_y), (128, 128, 128), 1)
        
        # Draw box
        cv2.rectangle(resize, (cdf_origin_x, cdf_graph_top), (cdf_origin_x + cdf_graph_width, cdf_origin_y), text_color, 2)
        
        # Draw target frametime reference line (vertical)
        target_x = cdf_origin_x + int((target_frametime / max_frametime_cdf) * cdf_graph_width)
        cv2.line(resize, (target_x, cdf_graph_top), (target_x, cdf_origin_y), (93, 232, 130), 1)
        
        # Labels
        draw_text_with_border(resize, "0%", (cdf_origin_x + cdf_graph_width + 10, cdf_origin_y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.3, text_color, 1, cv2.LINE_AA)
        draw_text_with_border(resize, "50%", (cdf_origin_x + cdf_graph_width + 10, median_y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.3, text_color, 1, cv2.LINE_AA)
        draw_text_with_border(resize, "100%", (cdf_origin_x + cdf_graph_width + 10, cdf_graph_top + 5), cv2.FONT_HERSHEY_DUPLEX, 0.3, text_color, 1, cv2.LINE_AA)
        
        # X-axis labels (frametime)
        draw_text_with_border(resize, "0", (cdf_origin_x - 5, cdf_origin_y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.3, text_color, 1, cv2.LINE_AA)
        draw_text_with_border(resize, f"{target_frametime:.0f}", (target_x - 8, cdf_origin_y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.3, (93, 232, 130), 1, cv2.LINE_AA)
        draw_text_with_border(resize, f"{max_frametime_cdf:.0f}", (cdf_origin_x + cdf_graph_width - 15, cdf_origin_y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.3, text_color, 1, cv2.LINE_AA)
        
        # Title
        draw_text_with_border(resize, "SMOOTHNESS (CDF)", (cdf_origin_x, cdf_graph_top - 15), cv2.FONT_HERSHEY_DUPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    plt_origin_x = 60
    plt_origin_y = 500
    pt_width = 0
    pt_spacing = 2
    pt_gap = pt_spacing - pt_width
    y_scale = 4

    graph_speed = 2 #rate at which graph is moving from right to left

    # Calculate FPS graph baseline and scale based on target_fps
    if target_fps == 60:
        fps_graph_max = 60  # Max FPS to display for 60fps mode
    else:  # 120fps mode
        fps_graph_max = 120  # Max FPS to display for 120fps mode
    
    # Define FPS graph bounds (leave 1 pixel margin at top so line is visible at peak)
    fps_graph_top = plt_origin_y + 1  # 1 pixel from top edge
    fps_graph_bottom = plt_origin_y + 40*y_scale
    
    for key, value in enumerate(fps_graph):
        # Start drawing after we have enough data (use 60 instead of target_fps)
        if count < fps_graph_samples - key + 60:
            continue
        if key == 0:
            continue
        prev_ft = fps_graph[key-1]
        
        # Clamp FPS values to display range
        value_clamped = max(0, min(value, fps_graph_max))
        prev_ft_clamped = max(0, min(prev_ft, fps_graph_max))
        
        # Calculate y positions - scale from 0 (bottom) to fps_graph_max (top with 1px margin)
        # Available height is (40*y_scale - 1) to leave room at top
        fps_pixel_scale = (40 * y_scale - 1) / fps_graph_max
        y_start = plt_origin_y + (40 * y_scale) - (prev_ft_clamped * fps_pixel_scale)
        y_end = plt_origin_y + (40 * y_scale) - (value_clamped * fps_pixel_scale)
        
        # Clamp to box boundaries
        y_start = max(fps_graph_top, min(fps_graph_bottom, y_start))
        y_end = max(fps_graph_top, min(fps_graph_bottom, y_end))
        
        start_point = plt_origin_x + key*graph_speed, int(y_start)
        end_point = plt_origin_x + (key+1)*graph_speed, int(y_end)
        cv2.line(resize, start_point, end_point, graph_color, 2)
    # cv2.rectangle(resize,(plt_origin_x+1,plt_origin_y+1),(plt_origin_x+fps_graph_samples*pt_spacing+1,plt_origin_y+40*y_scale+1),shadow_color,1)
    cv2.rectangle(resize,(plt_origin_x,plt_origin_y),(plt_origin_x+fps_graph_samples*pt_spacing,plt_origin_y+40*y_scale),text_color,2)
    
    # Draw reference lines at key FPS values
    if target_fps == 60:
        # For 60fps mode: 60fps is at top (y_scale proportional)
        # 30fps is at halfway (20*y_scale)
        midline_y_30 = plt_origin_y + 20*y_scale
        # No 60fps line needed as it's at the top edge
        cv2.line(resize,(plt_origin_x,midline_y_30),(plt_origin_x+fps_graph_samples*pt_spacing,midline_y_30),text_color,1)
    else:
        # For 120fps mode: 120fps at top, 60fps at halfway, 30fps at 3/4 down
        midline_y_60 = plt_origin_y + 20*y_scale  # 60fps is halfway
        midline_y_30 = plt_origin_y + 30*y_scale  # 30fps is 3/4 down
        cv2.line(resize,(plt_origin_x,midline_y_60),(plt_origin_x+fps_graph_samples*pt_spacing,midline_y_60),text_color,1)
        cv2.line(resize,(plt_origin_x,midline_y_30),(plt_origin_x+fps_graph_samples*pt_spacing,midline_y_30),text_color,1)
    
    # Set midline_y for label positioning (always use 30fps line)
    midline_y = plt_origin_y + (20*y_scale if target_fps == 60 else 30*y_scale)
    draw_text_with_border(resize, "FRAME-RATE (FPS)", (plt_origin_x+fps_graph_samples*pt_spacing -150,plt_origin_y-15), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
    # Always show all four labels: 120, 60, 30, 0
    # Calculate positions based on the graph scale
    if target_fps == 60:
        # 60fps mode: 60 is at top, scale is 0-60
        fps_60_y = plt_origin_y + 5
        fps_120_y = plt_origin_y - 15  # Above the box (not on scale)
    else:
        # 120fps mode: 120 is at top, 60 is halfway
        fps_120_y = plt_origin_y + 5
        fps_60_y = plt_origin_y + 20*y_scale + 5
    
    draw_text_with_border(resize, "120", (plt_origin_x+fps_graph_samples*pt_spacing+10, fps_120_y), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    draw_text_with_border(resize, "60", (plt_origin_x+fps_graph_samples*pt_spacing+10, fps_60_y), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    draw_text_with_border(resize, "30", (plt_origin_x+fps_graph_samples*pt_spacing+10,midline_y+5), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    draw_text_with_border(resize, "0", (plt_origin_x+fps_graph_samples*pt_spacing+10,plt_origin_y+41*y_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color, 1, cv2.LINE_AA)
        # if key !=0:
            # cv2.line(resize,(plt_origin_x + key*pt_spacing - pt_gap, plt_origin_y- prev_ft*10),(plt_origin_x + key*pt_spacing, plt_origin_y-value*10),graph_color,1)

    # Draw Frametime Histogram
    if len(frametime_history) > 10:  # Only draw if we have enough data
        hist_origin_x = 1000
        hist_origin_y = 420  # Match frametime graph Y position
        hist_width = 270  # Wider to prevent text overlap
        hist_height = 100  # Slightly smaller to fit better
        
        # Define frametime buckets based on target FPS
        # Ranges designed so perfect framerate falls in "Perfect" category
        if target_fps == 60:
            # 60fps target is 16.67ms - Perfect should include 16.67ms
            buckets = [
                (0, 18, "Perfect", (93, 232, 130)),      # Green - includes ideal 16.67ms
                (18, 25, "Good", (93, 232, 130)),        # Green - still smooth
                (25, 35, "Fair", (0, 255, 255)),         # Yellow - noticeable
                (35, 50, "Stutter", (0, 165, 255)),      # Orange - severe
                (50, 999, "Bad", (0, 0, 255))            # Red - critical
            ]
        else:  # 120fps
            # 120fps target is 8.33ms - Perfect should include 8.33ms
            buckets = [
                (0, 9, "Perfect", (93, 232, 130)),       # Green - includes ideal 8.33ms
                (9, 12, "Good", (93, 232, 130)),         # Green - still smooth
                (12, 17, "Fair", (0, 255, 255)),         # Yellow - noticeable
                (17, 25, "Stutter", (0, 165, 255)),      # Orange - severe
                (25, 999, "Bad", (0, 0, 255))            # Red - critical
            ]
        
        # Count frames in each bucket
        bucket_counts = [0] * len(buckets)
        for ft in frametime_history:
            for i, (min_ft, max_ft, label, color) in enumerate(buckets):
                if min_ft <= ft < max_ft:
                    bucket_counts[i] += 1
                    break
        
        # Calculate percentages
        total = sum(bucket_counts)
        bucket_percentages = [(count / total * 100) if total > 0 else 0 for count in bucket_counts]
        
        # Draw histogram background
        cv2.rectangle(resize, (hist_origin_x, hist_origin_y - hist_height), 
                     (hist_origin_x + hist_width, hist_origin_y), text_color, 2)
        
        # Calculate average frametime
        avg_frametime = sum(frametime_history) / len(frametime_history) if len(frametime_history) > 0 else 0
        
        # Determine color for average frametime based on buckets
        avg_ft_color = text_color
        for min_ft, max_ft, label, color in buckets:
            if min_ft <= avg_frametime < max_ft:
                avg_ft_color = color
                break
        
        # Calculate average FPS from fps_graph (using same window as histogram would be ideal)
        # Use the most recent portion of fps_graph that corresponds to histogram size
        recent_fps_samples = fps_graph[-min(len(fps_graph), histogram_sample_size):]
        avg_fps = sum(recent_fps_samples) / len(recent_fps_samples) if len(recent_fps_samples) > 0 else 0
        
        # Determine color for average FPS (green if near target, yellow/orange/red as it drops)
        if target_fps == 60:
            if avg_fps >= 58:
                avg_fps_color = (93, 232, 130)  # Green - near perfect
            elif avg_fps >= 50:
                avg_fps_color = (0, 255, 255)   # Yellow - noticeable
            elif avg_fps >= 40:
                avg_fps_color = (0, 165, 255)   # Orange - poor
            else:
                avg_fps_color = (0, 0, 255)     # Red - bad
        else:  # 120fps
            if avg_fps >= 115:
                avg_fps_color = (93, 232, 130)  # Green - near perfect
            elif avg_fps >= 100:
                avg_fps_color = (0, 255, 255)   # Yellow - noticeable
            elif avg_fps >= 80:
                avg_fps_color = (0, 165, 255)   # Orange - poor
            else:
                avg_fps_color = (0, 0, 255)     # Red - bad
        
        # Draw title
        draw_text_with_border(resize, "FRAMETIME DISTRIBUTION", 
                   (hist_origin_x, hist_origin_y - hist_height - 55),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        
        # Draw average frametime below title
        avg_ft_text = f"Frametime Avg: {avg_frametime:.1f}ms"
        draw_text_with_border(resize, avg_ft_text, 
                   (hist_origin_x, hist_origin_y - hist_height - 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, avg_ft_color, 1, cv2.LINE_AA)
        
        # Draw average FPS below frametime average
        avg_fps_text = f"FPS Average: {avg_fps:.1f}"
        draw_text_with_border(resize, avg_fps_text, 
                   (hist_origin_x, hist_origin_y - hist_height - 25),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, avg_fps_color, 1, cv2.LINE_AA)
        
        # Draw stutter statistics if detection enabled
        if detect_stutters and len(stutter_events) > 0:
            severe_count = sum(1 for _, _, stype in stutter_events if stype == "severe")
            moderate_count = sum(1 for _, _, stype in stutter_events if stype == "moderate")
            total_stutters = len(stutter_events)
            
            # Color code: Red if any severe, orange if only moderate
            stutter_color = (0, 0, 255) if severe_count > 0 else (0, 140, 255)
            
            stutter_text = f"Stutters: {total_stutters} ({severe_count} severe, {moderate_count} moderate)"
            draw_text_with_border(resize, stutter_text, 
                       (hist_origin_x, hist_origin_y - hist_height - 10),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, stutter_color, 1, cv2.LINE_AA)
        
        # Draw bars
        bar_width = hist_width // len(buckets)
        for i, (percentage, (min_ft, max_ft, label, color)) in enumerate(zip(bucket_percentages, buckets)):
            if percentage > 0:
                bar_height = int((percentage / 100) * hist_height)
                bar_x = hist_origin_x + i * bar_width
                bar_y = hist_origin_y - bar_height
                
                # Draw bar
                cv2.rectangle(resize, (bar_x + 2, bar_y), 
                             (bar_x + bar_width - 2, hist_origin_y - 2), color, -1)
                
                # Draw percentage on bar if significant
                if percentage > 5:
                    text = f"{percentage:.0f}%"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.3, 1)[0]
                    text_x = bar_x + (bar_width - text_size[0]) // 2
                    text_y = bar_y + 15
                    draw_text_with_border(resize, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw labels below histogram with more spacing
        label_y = hist_origin_y + 15
        for i, (min_ft, max_ft, label, color) in enumerate(buckets):
            label_x = hist_origin_x + i * bar_width + 3
            # Draw color indicator
            cv2.rectangle(resize, (label_x, label_y), (label_x + 6, label_y + 6), color, -1)
            # Draw label text
            draw_text_with_border(resize, label, (label_x + 10, label_y + 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.28, text_color, 1, cv2.LINE_AA)
        
        # Draw frametime range thresholds above percentages
        threshold_y = hist_origin_y + 30
        for i, (min_ft, max_ft, label, color) in enumerate(buckets):
            threshold_x = hist_origin_x + i * bar_width + 3
            # Format range text
            if max_ft >= 999:
                range_text = f"{min_ft:.0f}+"
            else:
                range_text = f"{min_ft:.0f}-{max_ft:.0f}"
            draw_text_with_border(resize, range_text, (threshold_x, threshold_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.25, text_color, 1, cv2.LINE_AA)
        
        # Draw percentage table below thresholds
        table_y = hist_origin_y + 45
        for i, (percentage, (min_ft, max_ft, label, color)) in enumerate(zip(bucket_percentages, buckets)):
            table_x = hist_origin_x + i * bar_width + 3
            pct_text = f"{percentage:.1f}%"
            draw_text_with_border(resize, pct_text, (table_x, table_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.28, text_color, 1, cv2.LINE_AA)


    # Calculate total time elapse since previous cv2.imshow, for PERFORMANCE ANALYSIS
    calc_time = (cv2.getTickCount() - timestamp_before_imshow)/ cv2.getTickFrequency()*1000
    # print calc_time
    if calc_time>target_frametime:


        perf_list.append((cv2.getTickCount() - timestamp_before_imshow)/ cv2.getTickFrequency()*1000-sum(perf_list))
        perf_list.append(calc_time)
        perf_list = list(np.around(np.array(perf_list),2))

        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")        
        print(perf_list)

      
    #vsync - sync to target FPS
    while ((cv2.getTickCount() - timestamp_before_imshow)/cv2.getTickFrequency()*10**6< 10**6/target_fps):
        continue
    # print (cv2.getTickCount() - timestamp_before_imshow)/cv2.getTickFrequency()*1000 #check if equal to target frametime, disable to ensure vsync works
    
    # Add title at top center if specified
    if title_text:
        h, w = resize.shape[:2]
        
        # Calculate text size for centering
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(title_text, font, font_scale, thickness)[0]
        
        # Center position at top
        text_x = (w - text_size[0]) // 2
        text_y = 40  # 40 pixels from top
        
        # Draw opaque background rectangle
        padding = 10
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_size[1] - padding
        bg_x2 = text_x + text_size[0] + padding
        bg_y2 = text_y + padding
        
        # Draw black background (fully opaque)
        cv2.rectangle(resize, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        # Draw title text with black border
        draw_text_with_border(resize, title_text, (text_x, text_y), 
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Add watermark to bottom left corner
    if watermark is not None:
        try:
            h, w = resize.shape[:2]
            wm_h, wm_w = watermark.shape[:2]
            
            # Position: bottom left with 10px margin
            x_offset = 10
            y_offset = h - wm_h - 10
            
            # Check if watermark has alpha channel
            if watermark.shape[2] == 4:  # RGBA
                # Extract alpha channel for transparency
                alpha = watermark[:, :, 3] / 255.0
                # Get BGR channels
                wm_bgr = watermark[:, :, :3]
                
                # Blend with transparency (50% opacity for subtle watermark)
                alpha = alpha * 0.5
                
                for c in range(3):
                    resize[y_offset:y_offset+wm_h, x_offset:x_offset+wm_w, c] = \
                        resize[y_offset:y_offset+wm_h, x_offset:x_offset+wm_w, c] * (1 - alpha) + \
                        wm_bgr[:, :, c] * alpha
            else:  # No alpha channel, just blend at 50% opacity
                overlay = resize[y_offset:y_offset+wm_h, x_offset:x_offset+wm_w].copy()
                cv2.addWeighted(watermark, 0.5, overlay, 0.5, 0, resize[y_offset:y_offset+wm_h, x_offset:x_offset+wm_w])
        except Exception as e:
            print(f"Warning: Could not apply watermark: {e}")
    
    timestamp_before_imshow = cv2.getTickCount()
    cv2.imshow('frame',resize)
    
    # Write frame to output video
    if out is not None:
        out.write(resize)
    
    # cv2.imshow('frame',frame_diff)
    # cv2.imshow('frame',ResizeWithAspectRatio(frame_diff,width=1280))
    perf_list =[(cv2.getTickCount() - timestamp_before_imshow)/ cv2.getTickFrequency()*1000]


    prev_frame = src_frame
    count = count + 1
    # frametime is incremented at the start of each loop, not here


    # if pause_frame:
       #  if cv2.waitKey(0) & 0xFF == ord('q'):
    #     	break
    # else:    
       #  if cv2.waitKey(1) & 0xFF == ord('q'):
       #      break
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    perf_list.append((cv2.getTickCount() - timestamp_before_imshow)/ cv2.getTickFrequency()*1000-sum(perf_list))

# Clean up
fvs.stop()
if out is not None:
    out.release()
    print(f"\nVideo saved successfully to: {output_path}")

# Export histogram data to CSV
if len(frametime_history) > 10:
    csv_path = os.path.splitext(file_path)[0] + "_histogram.csv"
    
    # Define buckets based on target FPS (same as display logic)
    if target_fps == 60:
        buckets = [
            (0, 18, "Perfect", (93, 232, 130)),
            (18, 25, "Good", (93, 232, 130)),
            (25, 35, "Fair", (0, 255, 255)),
            (35, 50, "Stutter", (0, 165, 255)),
            (50, 999, "Bad", (0, 0, 255))
        ]
    else:  # 120fps
        buckets = [
            (0, 9, "Perfect", (93, 232, 130)),
            (9, 12, "Good", (93, 232, 130)),
            (12, 17, "Fair", (0, 255, 255)),
            (17, 25, "Stutter", (0, 165, 255)),
            (25, 999, "Bad", (0, 0, 255))
        ]
    
    # Count frames in each bucket
    bucket_counts = [0] * len(buckets)
    for ft in frametime_history:
        for i, (min_ft, max_ft, label, color) in enumerate(buckets):
            if min_ft <= ft < max_ft:
                bucket_counts[i] += 1
                break
    
    # Calculate percentages
    total = sum(bucket_counts)
    bucket_percentages = [(count / total * 100) if total > 0 else 0 for count in bucket_counts]
    
    # Calculate averages
    avg_frametime = sum(frametime_history) / len(frametime_history) if len(frametime_history) > 0 else 0
    avg_fps = sum(fps_graph[-min(len(fps_graph), histogram_sample_size):]) / min(len(fps_graph), histogram_sample_size) if len(fps_graph) > 0 else 0
    
    # Calculate actual duration based on unique frames analyzed
    actual_seconds = total / avg_fps if avg_fps > 0 else 0
    actual_minutes = actual_seconds / 60
    
    # Write CSV
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header with metadata
            writer.writerow(['Frametime Analysis Results'])
            writer.writerow(['Video File', file_path])
            writer.writerow(['Target FPS', target_fps])
            writer.writerow(['Average Frametime (ms)', f'{avg_frametime:.2f}'])
            writer.writerow(['Average FPS', f'{avg_fps:.2f}'])
            writer.writerow(['Total Unique Frames Analyzed', total])
            writer.writerow(['Analysis Duration', f'{actual_seconds:.1f} seconds ({actual_minutes:.2f} minutes)'])
            writer.writerow(['Sample Window (Max)', f'{histogram_sample_size} frames (last 30 minutes of video)'])
            writer.writerow([])  # Empty row
            
            # Write stutter detection data if enabled
            if detect_stutters:
                severe_count = sum(1 for _, _, stype in stutter_events if stype == "severe")
                moderate_count = sum(1 for _, _, stype in stutter_events if stype == "moderate")
                total_stutters = len(stutter_events)
                
                writer.writerow(['Stutter Detection Results'])
                writer.writerow(['Total Stutters Detected', total_stutters])
                writer.writerow(['Severe Stutters (>= 4x target frametime)', severe_count])
                writer.writerow(['Moderate Stutters (>= 2.5x target frametime)', moderate_count])
                writer.writerow(['Severe Threshold (ms)', f'{target_frametime * severe_stutter_threshold_multiplier:.1f}'])
                writer.writerow(['Moderate Threshold (ms)', f'{target_frametime * stutter_threshold_multiplier:.1f}'])
                
                if total_stutters > 0:
                    # Calculate average and worst stutter severity
                    stutter_frametimes = [ft for _, ft, _ in stutter_events]
                    avg_stutter_ft = sum(stutter_frametimes) / len(stutter_frametimes)
                    worst_stutter_ft = max(stutter_frametimes)
                    
                    writer.writerow(['Average Stutter Frametime (ms)', f'{avg_stutter_ft:.1f}'])
                    writer.writerow(['Worst Stutter Frametime (ms)', f'{worst_stutter_ft:.1f}'])
                
                writer.writerow([])  # Empty row
            
            # Write bucket data
            writer.writerow(['Bucket', 'Range (ms)', 'Frame Count', 'Percentage'])
            for i, (min_ft, max_ft, label, color) in enumerate(buckets):
                if max_ft >= 999:
                    range_text = f'{min_ft:.0f}+'
                else:
                    range_text = f'{min_ft:.0f}-{max_ft:.0f}'
                writer.writerow([label, range_text, bucket_counts[i], f'{bucket_percentages[i]:.2f}%'])
        
        print(f"Histogram data exported to: {csv_path}")
    except Exception as e:
        print(f"Warning: Could not export CSV: {e}")

cv2.destroyAllWindows()