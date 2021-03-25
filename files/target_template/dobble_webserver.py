from flask import Flask,render_template,request,make_response,Response
import numpy as np
import cv2
import base64
import sys
import json
import os
import itertools

global vart_present
global plot_present
vart_present = False
plot_present = False

global display_fps
display_fps = False

global circle_minRadius
global circle_maxRadius
circle_minRadius = 100
circle_maxRadius = 200

try:
   import vart
   import pathlib
   import xir
   import runner
   vart_present = True
except:
   print("[ERROR] VART not available ! (are you running on a Vitis-AI 1.3 platform ?)")
   

try:
   import sensors
   import time
   import matplotlib.pyplot as plt
   from matplotlib.figure import Figure
   from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
   import io
   plot_present = True
except:
   print("[ERROR] sensor/matplotlib/io not available ! (install with : pip3 install pysensors matplotlib io)")


#if vart_present == True:
#   sys.path.append(os.path.abspath('../'))
#   sys.path.append(os.path.abspath('./'))
#   from vitis_ai_vart.facedetect import FaceDetect
#   from vitis_ai_vart.facelandmark import FaceLandmark
#   from vitis_ai_vart.utils import get_child_subgraph_dpu


def get_subgraph (g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    return sub

"""
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
"""
import math
def CPUCalcSoftmax(data, size):
    sum = 0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i])
        sum += result[i]
    for i in range(size):
        result[i] /= sum
    return result

"""
Get topk results according to its probability
datain: data result of softmax
filePath: filePath in witch that records the infotmation of kinds
"""

def TopK(datain, size, filePath):

    cnt = [i for i in range(size)]
    pair = zip(datain, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    for i in range(5):
        idx = 0
        for line in data1:
            if idx == cnt_new[i]:
                print("Top[%d] %d %s" % (i, idx, (line.strip)("\n")))
            idx = idx + 1



import dobble_utils as db


# Define App
app = Flask(__name__,template_folder="templates")

# The home page is routed to index.html inside
@app.route('/')
def index():
   return render_template('index.html')


@app.route('/set_threshold/<slider>/<value>',methods=["POST"])
def set_threshold(slider,value):
   global circle_minRadius
   global circle_maxRadius

   #print("[INFO] set_threshold ", slider, " ", value)

   if slider == "min":
      circle_minRadius = int(value)
      print("[INFO] circle_minRadius = ",circle_minRadius)

   if slider == "max":
      circle_maxRadius = int(value)
      print("[INFO] circle_maxRadius = ",circle_maxRadius)

   # Return result as a json object
   return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/set_fps_option/<value>',methods=["POST"])
def set_fps_option(value):
   global display_fps

   if value == "true": 
      display_fps = True
      print("[INFO] display FPS = ON")
   if value == "false": 
      display_fps = False
      print("[INFO] display FPS = OFF")

   # Return result as a json object
   return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

def generate():
  global vart_present
  global display_fps
  global circle_minRadius
  global circle_maxRadius

  #if vart_present == True:
  
  # Parameters (tweaked for video)
  dir = './dobble_dataset'

  scale = 1.0
  circle_minRadius = int(100*scale)
  circle_maxRadius = int(200*scale)
  circle_xxxRadius = int(250*scale)

  b = int(4*scale) # border around circle for bounding box

  text_fontType = cv2.FONT_HERSHEY_SIMPLEX
  text_fontSize = 0.75*scale
  text_color    = (0,0,255)
  text_lineSize = max( 1, int(2*scale) )
  text_lineType = cv2.LINE_AA

  matching_x = int(10*scale)
  matching_y = int(20*scale)

  input_video = 0 # laptop camera
  #input_video = 1 # USB webcam

  displayReference = True

  # Open video
  cap = cv2.VideoCapture(input_video)
  frame_width = 640
  frame_height = 480
  cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
  #frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
  #frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  print("camera",input_video," (",frame_width,",",frame_height,")")

  # Open dobble model
  #model = load_model('dobble_model.h5')

  # Vitis-AI implemenation of dobble model


  # Create DPU runner
  g = xir.Graph.deserialize('model_dir/dobble.xmodel')
  subgraphs = get_subgraph (g)
  assert len(subgraphs) == 1 # only one DPU kernel
  dpu = runner.Runner(subgraphs[0],"run")

  # Get input/output tensors
  inputTensors = dpu.get_input_tensors()
  outputTensors = dpu.get_output_tensors()
  inputShape = inputTensors[0].dims
  outputShape = outputTensors[0].dims

  # Load reference images
  train1_dir = dir+'/dobble_deck01_cards_57'
  train1_cards = db.capture_card_filenames(train1_dir)
  train1_X,train1_y = db.read_and_process_image(train1_cards,72,72)

  # Load mapping/symbol databases
  symbols = db.load_symbol_labels(dir+'/dobble_symbols.txt')
  mapping = db.load_card_symbol_mapping(dir+'/dobble_card_symbol_mapping.txt')

  image = []
  output = []
  circle_list = []
  bbox_list = []
  card_list = []

  frame_count = 0

  # init the real-time FPS counter
  rt_fps_count = 0
  rt_fps_time = cv2.getTickCount()
  rt_fps_valid = False
  rt_fps = 0.0
  rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
  rt_fps_x = int(10*scale)
  rt_fps_y = int((frame_height-10)*scale)
    
  while True:
    # init the real-time FPS counter
    if rt_fps_count == 0:
        rt_fps_time = cv2.getTickCount()

    #if cap.grab():
    if True:
        frame_count = frame_count + 1
        #flag, image = cap.retrieve()
        flag, image = cap.read()
        if not flag:
            break
        else:
            image = cv2.resize(image,(0,0), fx=scale, fy=scale) 
            output = image.copy()
            
            # detect circles in the image
            gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.medianBlur(gray1,5)
            circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1.5 , 100, minRadius=circle_minRadius,maxRadius=circle_maxRadius)

            circle_list = []
            bbox_list = []
            card_list = []
            
            # ensure at least some circles were found
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                # loop over the (x, y) coordinates and radius of the circles
                for (cx, cy, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
                    #cv2.rectangle(output, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 128, 255), -1)

                    # extract ROI for card
                    y1 = (cy-r-b)
                    y2 = (cy+r+b)
                    x1 = (cx-r-b)
                    x2 = (cx+r+b)
                    roi = output[ y1:y2, x1:x2, : ]
                    cv2.rectangle(output, (x1,y1), (x2,y2), (0, 0, 255), 2)
                    
                    try:
                        # dobble pre-processing
                        card_img = cv2.resize(roi,(224,224),interpolation=cv2.INTER_CUBIC)
                        card_img = card_img/255.0
                        card_x = []
                        card_x.append( card_img )
                        card_x = np.array(card_x)

                        # dobble model execution
                        #card_y = model.predict(card_x)

                        """ Prepare input/output buffers """
                        #print("[INFO] process - prep input buffer ")
                        inputData = []
                        inputData.append(np.empty((inputShape),dtype=np.float32,order='C'))
                        inputImage = inputData[0]
                        inputImage[0,...] = card_img

                        #print("[INFO] process - prep output buffer ")
                        outputData = []
                        outputData.append(np.empty((outputShape),dtype=np.float32,order='C'))

                        """ Execute model on DPU """
                        #print("[INFO] process - execute ")
                        job_id = dpu.execute_async( inputData, outputData )
                        dpu.wait(job_id)

                        # dobble post-processing
                        if False:
                            softmax = CPUCalcSoftmax(outputData[0][0], 58)
                            #TopK(softmax, 58, "./words.txt")
                            card_y = softmax
                        else:
                            OutputData = outputData[0].reshape(1,58)
                            card_y = np.reshape( OutputData, (-1,58) )
                        #print(card_y)

                        card_id  = np.argmax(card_y[0])
                        cv2.putText(output,str(card_id),(x1,y1-b),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)
                        
                        # Add ROI to card/bbox lists
                        if card_id > 0:
                            circle_list.append((cx,cy,r))
                            bbox_list.append((x1,y1,x2,y2))
                            card_list.append(card_id)

                            if displayReference:
                                reference_img = train1_X[card_id-1]
                                reference_shape = reference_img.shape
                                reference_x = reference_shape[0]
                                reference_y = reference_shape[1]
                                output[y1:y1+reference_y,x1:x1+reference_x,:] = reference_img
                        
                    except:
                        print("ERROR : Exception occured during dobble classification ...")

                         
            if len(card_list) == 1:
                matching_text = ("[%04d] %02d"%(frame_count,card_list[0]))
                #print(matching_text)
                
            if len(card_list) > 1:
                #print(card_list)
                matching_text = ("[%04d]"%(frame_count))
                for card_pair in itertools.combinations(card_list,2):
                    #print("\t",card_pair)
                    card1_mapping = mapping[card_pair[0]]
                    card2_mapping = mapping[card_pair[1]]
                    symbol_ids = np.intersect1d(card1_mapping,card2_mapping)
                    #print("\t",symbol_ids)
                    symbol_id = symbol_ids[0]
                    symbol_label = symbols[symbol_id]
                    #print("\t",symbol_id," => ",symbol_label)
                    matching_text = matching_text + (" %02d,%02d=%s"%(card_pair[0],card_pair[1],symbol_label) )
                #print(matching_text)
                cv2.putText(output,matching_text,(matching_x,matching_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)                

    # real-time FPS display
    if display_fps == True and rt_fps_valid == True:
        cv2.putText(output, rt_fps_message, (rt_fps_x,rt_fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # Encode video frame to JPG
    (flag, encodedImage) = cv2.imencode(".jpg", output)
    if not flag:
        continue
      
    # yield the output frame in the byte format
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    # Update the real-time FPS counter
    rt_fps_count = rt_fps_count + 1
    if rt_fps_count >= 10:
        t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
        rt_fps_valid = True
        rt_fps = 10.0/t
        rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
        #print("[INFO] ",rt_fps_message)
        rt_fps_count = 0

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
   return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


def monitor():

  if plot_present == True:
   sensors.init()
   
   pmbus_ultra96v2 = {
      # ir38060-i2c-6-45
      "ir38060-i2c-6-45:pout1" : "5V",
      # irps5401-i2c-6-43
      "irps5401-i2c-6-43:pout1" : "VCCAUX",
      "irps5401-i2c-6-43:pout2" : "VCCO 1.2V",
      "irps5401-i2c-6-43:pout3" : "VCCO 1.1V",
      "irps5401-i2c-6-43:pout4" : "VCCINT",
      "irps5401-i2c-6-43:pout5" : "3.3V DP",
      # irps5401-i2c-6-44
      "irps5401-i2c-6-44:pout1" : "VCCPSAUX",
      "irps5401-i2c-6-44:pout2" : "PSINT_LP",
      "irps5401-i2c-6-44:pout3" : "VCCO 3.3V",
      "irps5401-i2c-6-44:pout4" : "PSINT_FP",
      "irps5401-i2c-6-44:pout5" : "PSPLL 1.2V",
   }

   pmbus_uz7ev_evcc = {
      # ir38063-i2c-3-4c
      "ir38063-i2c-3-4c:pout1" : "Carrier 3V3",
      # ir38063-i2c-3-4b
      "ir38063-i2c-3-4b:pout1" : "Carrier 1V8",
      # irps5401-i2c-3-4a
      "irps5401-i2c-3-4a:pout1" : "Carrier 0V9 MGTAVCC",
      "irps5401-i2c-3-4a:pout2" : "Carrier 1V2 MGTAVTT",
      "irps5401-i2c-3-4a:pout3" : "Carrier 1V1 HDMI",
      "irps5401-i2c-3-4a:pout4" : "Unused",
      "irps5401-i2c-3-4a:pout5" : "Carrier 1V8 MGTVCCAUX LDO",
      # irps5401-i2c-3-49
      "irps5401-i2c-3-49:pout1" : "Carrier 0V85 MGTRAVCC",
      "irps5401-i2c-3-49:pout2" : "Carrier 1V8 VCCO",
      "irps5401-i2c-3-49:pout3" : "Carrier 3V3 VCCO",
      "irps5401-i2c-3-49:pout4" : "Carrier 5V MAIN",
      "irps5401-i2c-3-49:pout5" : "Carrier 1V8 MGTRAVTT LDO",
      # ir38063-i2c-3-48
      "ir38063-i2c-3-48:pout1" : "SOM 0V85 VCCINT",
      # irps5401-i2c-3-47
      "irps5401-i2c-3-47:pout1" : "SOM 1V8 VCCAUX",
      "irps5401-i2c-3-47:pout2" : "SOM 3V3",
      "irps5401-i2c-3-47:pout3" : "SOM 0V9 VCUINT",
      "irps5401-i2c-3-47:pout4" : "SOM 1V2 VCCO_HP_66",
      "irps5401-i2c-3-47:pout5" : "SOM 1V8 PSDDR_PLL LDO",
      # irps5401-i2c-3-46
      "irps5401-i2c-3-46:pout1" : "SOM 1V2 VCCO_PSIO",
      "irps5401-i2c-3-46:pout2" : "SOM 0V85 VCC_PSINTLP",
      "irps5401-i2c-3-46:pout3" : "SOM 1V2 VCCO_PSDDR4_504",
      "irps5401-i2c-3-46:pout4" : "SOM 0V85 VCC_PSINTFP",
      "irps5401-i2c-3-46:pout5" : "SOM 1V2 VCC_PSPLL LDO",
   }

   pmbus_uz3eg_xxx = {
      # irps5401-i2c-3-43
      "irps5401-i2c-3-43:pout1" : "PSIO",
      "irps5401-i2c-3-43:pout2" : "VCCAUX",
      "irps5401-i2c-3-43:pout3" : "PSINTLP",
      "irps5401-i2c-3-43:pout4" : "PSINTFP",
      "irps5401-i2c-3-43:pout5" : "PSPLL",
      # irps5401-i2c-3-44
      "irps5401-i2c-3-44:pout1" : "PSDDR4",
      "irps5401-i2c-3-44:pout2" : "INT_IO",
      "irps5401-i2c-3-44:pout3" : "3.3V",
      "irps5401-i2c-3-44:pout4" : "INT",
      "irps5401-i2c-3-44:pout5" : "PSDDRPLL",
      # irps5401-i2c-3-45
      "irps5401-i2c-3-45:pout1" : "MGTAVCC",
      "irps5401-i2c-3-45:pout2" : "5V",
      "irps5401-i2c-3-45:pout3" : "3.3V",
      "irps5401-i2c-3-45:pout4" : "VCCO 1.8V",
      "irps5401-i2c-3-45:pout5" : "MGTAVTT",
   }

   pmbus_annotations = {
      "ULTRA96V2"   : pmbus_ultra96v2,
      "UZ7EV_EVCC"  : pmbus_uz7ev_evcc,
      "UZ3EG_IOCC"  : pmbus_uz3eg_xxx,
      "UZ3EG_PCIEC" : pmbus_uz3eg_xxx
   }

   use_annotations = 1
   target = "ULTRA96V2"
   target_annotations = pmbus_annotations[target]

   pmbus_power_features = []
   for chip in sensors.iter_detected_chips():
      device_name = str(chip)
      adapter_name = str(chip.adapter_name)
      #print( "%s at %s" % (device_name, adapter_name) )
      #if 'irps5401' in device_name:
      #if 'ir38063' in device_name:
      if True:
         for feature in chip:
            feature_name = str(feature.label)
            if 'pout' in feature_name:
               label = device_name + ":" + feature_name
               pmbus_power_features.append( (label, feature) )
               feature_value = feature.get_value() * 1000.0
               print( " %s : %8.3f mW" % (feature_name, feature_value) )

   N = len(pmbus_power_features)
   p_x = np.linspace(1,N,N)
   p_y = np.linspace(0,0,N,dtype=float)

   p_l = []
   for (i, (label,feature)) in enumerate(pmbus_power_features):
      p_l.append(label)
   if use_annotations == 1:
      for i in range( len(p_l) ):
         label = p_l[i]
         if label in target_annotations:
            label = target_annotations[label]
            p_l[i] = label


   fig = plt.figure()
   a = fig.add_subplot(1,1,1)
   a.barh(p_x,p_y) 

   while True:
      for (i, (label,feature)) in enumerate(pmbus_power_features):
         p_y[i] = feature.get_value() * 1000.0

      fig.delaxes(a)
      a = fig.add_subplot(1,1,1)
      a.barh(p_x,p_y)
      a.set_xlabel('Power (mW)')
      a.set_xlim(xmin=0.0,xmax=10000.0)
      a.set_yticks(p_x)
      a.set_yticklabels(p_l)

      fig.canvas.draw()

      plot_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
      plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      plot_img = cv2.cvtColor(plot_img,cv2.COLOR_RGB2BGR)

      # Encode video frame to JPG
      (plot_flag, plot_encodedImage) = cv2.imencode(".jpg", plot_img)
      if not plot_flag:
         continue
      
      # yield the output frame in the byte format
      yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(plot_encodedImage) + b'\r\n')

@app.route("/power_feed")
def power_feed():
   # return the response generated along with the specific media
   # type (mime type)
   return Response(monitor(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
   #app.run(debug = True)
   app.run(host='0.0.0.0', port=80, debug=True)

