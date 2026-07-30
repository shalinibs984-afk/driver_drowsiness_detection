import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np
import av
#from pygame import mixer

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Start the webcam and look at the camera.")

# Alarm
#mixer.init()
#try:
    #mixer.music.load("alarm.mpeg")
#except:
    #st.warning("alarm.mpeg not found.")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.LEFT = [33, 160, 158, 133, 153, 144]
        self.RIGHT = [362, 385, 387, 263, 373, 380]

        self.closed_frames = 0
        self.threshold = 0.22

    def ear(self, pts):
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])

        return (A + B) / (2 * C)

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if results.multi_face_landmarks:

            face = results.multi_face_landmarks[0]

            h, w, _ = img.shape

            left = np.array(
                [[face.landmark[i].x * w, face.landmark[i].y * h] for i in self.LEFT]
            )

            right = np.array(
                [[face.landmark[i].x * w, face.landmark[i].y * h] for i in self.RIGHT]
            )

            leftEAR = self.ear(left)
            rightEAR = self.ear(right)

            avgEAR = (leftEAR + rightEAR) / 2

            cv2.putText(
                img,
                f"EAR: {avgEAR:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if avgEAR < self.threshold:

                self.closed_frames += 1

                cv2.putText(
                    img,
                    "DROWSY",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

                if self.closed_frames > 15:
                    if not mixer.music.get_busy():
                        mixer.music.play()

            else:

                self.closed_frames = 0
                mixer.music.stop()

                cv2.putText(
                    img,
                    "AWAKE",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)


    webrtc_streamer(
    key="driver",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15},
            "facingMode": "user",
        },
        "audio": False,
    },
    async_processing=True,
            )
