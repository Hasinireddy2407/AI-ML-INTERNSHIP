# Day 4 — Task 1: Improve Pose Detection Accuracy
##  Pose Detection

Pose detection is a computer vision technique that identifies and tracks the position of human body parts in an image or video. It works by detecting key points on the body called landmarks such as shoulders, elbows, wrists, hips, knees, and ankles. These landmarks are then connected to form a skeleton-like structure that represents the human pose.

## Why Accuracy Needs Improvement

Out of the box, MediaPipe works well but there are situations where it struggles. Poor lighting causes the model to miss landmarks or assign low visibility scores. If the person is too far from the camera or partially out of frame, fewer landmarks will be detected. A cluttered background can confuse the detector. The default settings use a balanced mode that prioritises speed over accuracy, so switching to accuracy mode gives better results.

## How to Improve Accuracy

-The most important change is setting model complexity to `2` instead of the default `0`. This uses the most powerful version of the model and detects landmarks more precisely especially on difficult images.

-Raising the detection confidence threshold to `0.7` means the model only reports a detection when it is at least 70 percent confident, reducing false detections.

-Good lighting matters a lot. Even, bright front-facing light helps the model see body outlines clearly. Avoid backlight or harsh shadows.

-Camera position should be straight-on and at full body height if possible. Side angles or extreme top-down angles reduce accuracy significantly.

-A plain or contrasting background helps the model distinguish the person from the surroundings more easily.


## Key Landmarks for Virtual Try-On

Not all 33 landmarks are needed for clothing overlay. The most important ones are the left and right shoulders, left and right elbows, left and right wrists, and left and right hips. The distance between the two shoulder landmarks in pixels is called the shoulder width and is used to scale the garment to the correct size for that person.


##  Output Tells You

After running pose detection you get the pixel coordinates of each detected landmark, the visibility score for each one, and the shoulder width measurement. If visibility scores for the shoulders are above `0.9` the detection is very reliable. If they are below `0.5` the garment overlay will likely be inaccurate and the image should be retaken with better conditions.



