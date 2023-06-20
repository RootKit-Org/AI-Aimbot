# Explain your model

- Tell the community about your model.         
Default yolov5n model, modified so that it works on 160x160 images.

- What data was it trained on?         
General images

- How much data was it trained on?
More info here: 
https://learnopencv.com/custom-object-detection-training-using-yolov5/#Training-the-Small-Model

- How many models do you have?       
One model

- Are they for pytorch, onnx, tensorrt, something else?
onnx

- Any set up info
You should replace this model with the one you are currently using. 
In addition, the size of the image captured and passed to the model should be changed from 320 to 160.
Works best with the 'main_onnx_amd_perf.py' script