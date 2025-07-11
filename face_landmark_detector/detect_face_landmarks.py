import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

#config
model_path = 'face_landmarker.task'
input_video = '/Users/felipepesantez/Movies/inference/person_facetr.mp4'
output_video = 'output.mp4'
landmarks_file = '3d_landmarks.npy'

#landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = 1,
    output_face_blendshapes = True,
    output_facial_transformation_matrixes = True
)

def draw_landmarks_on_image(rgb_image, detection_result):
    annotaded_image = np.copy(rgb_image)

    for face_landmarks in detection_result.face_landmarks:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotaded_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style() 
        )

    return annotaded_image

#video
all_landmarks = []

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

with FaceLandmarker.create_from_options(options) as landmarker:
    frame_timestamps_ms = 0

    while cap.isOpened():
        sucess, frame = cap.read()
        if not sucess:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = landmarker.detect_for_video(mp_image, frame_timestamps_ms)

        if detection_result.face_landmarks:
            annoted_image = draw_landmarks_on_image(rgb_frame, detection_result)
            output_frame = cv2.cvtColor(annoted_image, cv2.COLOR_RGB2BGR)

            if detection_result.face_landmarks:
                world_landmarks = []
                for landmark in detection_result.face_landmarks[0]:
                    world_landmarks.append([landmark.x, landmark.y, landmark.z])
                all_landmarks.append(world_landmarks)

        else:
            output_frame = frame

        out.write(output_frame)
        frame_timestamps_ms += int(1000 / fps)

        cv2.imshow('Face Mesh', output_frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()

if all_landmarks:
    np.save(landmarks_file, np.array(all_landmarks))
    print("Saved 3d landmarks")

print("Process complete!")