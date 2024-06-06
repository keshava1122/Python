import cv2
import numpy as np
import time


class Yolov4:
    def __init__(self):
        self.weights = '/home/zestiot/Desktop/Zestiot/PROJECTS/Bubble_Detection/cylinder_count_model/defect_model/M872_C4_B128_S8_I10000_best.weights'
        self.cfg = '/home/zestiot/Desktop/Zestiot/PROJECTS/Bubble_Detection/cylinder_count_model/defect_model/M872_C4_B128_S8_I10000.cfg'
        self.classes = ["Cylinder_cap", "Vpring", "Bend_Vpring", "Cylinder_Digs", "Bend_Stayplates", "No_cap"]
        self.Neural_Network = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
        self.outputs = self.Neural_Network.getUnconnectedOutLayersNames()
        self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        self.image_size = 320
        self.vpring_count = 0
        self.vpring_objects = {}  # Dictionary to store unique IDs for Vpring objects

    def bounding_box(self, detections):
        print(detections[0].shape)
        confidence_score = []
        ids = []
        cordinates = []
        Threshold = 0.5
        for i in detections:
            for j in i:
                probs_values = j[5:]
                class_ = np.argmax(probs_values)
                confidence_ = probs_values[class_]

                if confidence_ > Threshold:
                    w, h = int(j[2] * self.image_size), int(j[3] * self.image_size)
                    x, y = int(j[0] * self.image_size - w / 2), int(j[1] * self.image_size - h / 2)
                    cordinates.append([x, y, w, h])
                    ids.append(class_)
                    confidence_score.append(float(confidence_))
        final_box = cv2.dnn.NMSBoxes(cordinates, confidence_score, Threshold, .6)
        return final_box, cordinates, confidence_score, ids

    def predictions(self, prediction_box, bounding_box, confidence, class_labels, width_ratio, height_ratio, end_time,
                    image):
        vpring_count_text = f"Vpring Count: {self.vpring_count}"  # Text for displaying Vpring count
        for i in range(len(prediction_box)):
            if i==0:
                continue
            print(i, prediction_box)
            for j in prediction_box[i]:
                x, y, w, h = bounding_box[j]
                x = int(x * width_ratio)
                y = int(y * height_ratio)
                w = int(w * width_ratio)
                h = int(h * height_ratio)
                label = str(self.classes[class_labels[j]])
                conf_ = str(round(confidence[j], 2))
                color = [int(c) for c in self.COLORS[class_labels[j]]]

                # If detected object is Vpring
                if label == "Vpring":
                    # Check if object already has an ID, if not, assign a new one
                    if j not in self.vpring_objects:
                        self.vpring_count += 1
                        self.vpring_objects[j] = self.vpring_count
                    label += f"_{self.vpring_objects[j]}"  # Append unique ID to label

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label + ' ' + conf_, (x, y - 2), cv2.FONT_HERSHEY_COMPLEX, .5, color, 2)
                time_str = f"Inference time: {end_time:.3f}"
                cv2.putText(image, time_str, (10, 13), cv2.FONT_HERSHEY_COMPLEX, .5, (156, 0, 166), 1)

        # Display Vpring count on the image
        cv2.putText(image, vpring_count_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0), 1)

        return image

    def Inference(self, image, original_width, original_height):
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (320, 320), True, crop=False)
        self.Neural_Network.setInput(blob)
        start_time = time.time()
        output_data = self.Neural_Network.forward(self.outputs)
        end_time = time.time() - start_time
        final_box, cordinates, confidence_score, ids = self.bounding_box(output_data)
        print("cordinates:", cordinates)
        outcome = self.predictions(final_box, cordinates, confidence_score, ids, original_width / 320,
                                   original_height / 320, end_time, image)
        return outcome


if __name__ == "__main__":
    obj = Yolov4()
    cap = cv2.VideoCapture('/home/zestiot/Downloads/Video_3.avi')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = '/home/zestiot/Desktop/Zestiot/PROJECTS/Bubble_Detection/Outputs/26.03.24/demo_1.04.24.avi'
    output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        res, frame = cap.read()
        if res:
            img = frame
            outcome = obj.Inference(image=frame, original_width=width, original_height=height)
            cv2.imshow("demo", outcome)
            output.write(outcome)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
