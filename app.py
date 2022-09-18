from helper import *

import streamlit as st


#from streamlit_webrtc import webrtc_streamer , RTCConfiguration



#class VideoProcessor:
#    def recv(self, frame) : 
#        #frm= frame.to_ndarray(format="bgr24") 
        
#        frm= predictf(frame)

#        return av.VideoFrame.from_ndarray(frm, format='bgr24') 

#webrtc_streamer (key="key", video_processor_factory= VideoProcessor , 
#rtc_configuration=RTCConfiguration ( {"iceServers" : [{"urls" : ["stun : stun.l.google.com : 19302"]}]} ) )



st.title('Hello to my website using streamlit')
st.header('It is a Face Recognition model that classify two different classes : me , Not_me')


#if st.button('START capturing video') and not st.button('STOP capturing video'):

container= st.container()
col1, col2 = container.columns(2)
#col3, col4 = container.columns([4,2])


h=col1.button('START capturing the video')
hh=col2.button('STOP capturing the video!')
video_capture = cv.VideoCapture(0)  # webcamera

#run=st.checkbox('Run') 
FRAME_WINDOW=container.image([]) 

#st.text('Predictions :-') 

#frame_text=col4.text([])


while h and not hh: #run: #and i<100:
    ret, frame = video_capture.read()
    if not ret:
        print("failed to grab frame")
    else:
        f =predictor(frame)
        FRAME_WINDOW.image(f,clamp=True, channels='BGR',caption='This is the classified person from webcam') 


