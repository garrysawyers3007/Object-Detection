# IMAGE AI
Detect objects in an image using ImageAi library.
The server takes image input from the user as JSON request as a base64 encoded and decodes and stores it and uses the Image Ai API to detect objects as specified by the user. The model used is YOLO which can detect upto 80 objects in the image.
The specifics of detection like speed, threshold probability, objects to be detected can be set by the user. 
The API gives an output image showing bounding boxes, class and probability detected by the model along with a dictionary consisting of details associated with the detected objects. The server returns this dicitonary as response in JSON format.
