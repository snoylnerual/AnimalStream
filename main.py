from ultralytics import YOLO
import numpy as np
import json
import os

# meta holds the names of all the videos and the objects seen within each video, plus the specific frames
with open('data/train/train/meta.json') as f:
    meta = json.load(f)
# meta_instance holds the names of all the videos and the objects seen within each video
with open('data/train/train/meta_instance.json') as f:
    meta_instance = json.load(f)

# model
# YOLO model yolov8s-seg.pt
model = YOLO("yolo-Weights/yolov8s-seg.pt")

# object classes for COCO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# class_Names = [model.names[n] for n in model.names]

values_not_in_class_Names = []

# creates a list of all the class labels in youtube_vis train dataset
youtube_vis_class_Names = []
for vids in meta_instance['videos']:
    for objs in meta_instance['videos'][vids]['objects']:
        c = meta_instance['videos'][vids]['objects'][objs]['category']
        if c not in youtube_vis_class_Names:
            youtube_vis_class_Names.append(c)

# gets the overlapping list of class labels between youtube_Vis and COCO
both_class_Names = [value for value in classNames if value in youtube_vis_class_Names]

# holds the precision and recall for each class
class_dic = {}
for c in both_class_Names:
    class_dic[c] = {'AP': [],
                    'Recall': []}

# iterates through all the folders in this directory
# each folder is a 'video', the folders hold each frame in jpeg form
directory = "data/train/train/JPEGImages"
for folder in os.listdir(directory):
    # predicts segemntation results of each frame with certain confidence and iou
    # stream = True allows for the model to see all the frames as one video instead of individuals jpegs
    video_results = model(('data/train/train/JPEGImages/'+folder), conf=0.4, iou=0.5, stream=True)

    # the model produces a generator of results objects, so need to convert to list form for each access
    # there is a results objects for each frame
    frame_results = [item for item in video_results]
    # for each frame, truth_counter_dict holds the number of instances of each class present
    # same form as predict_counter_dic
    truth_counter_dict = {}
    for d in meta['videos'][folder]['objects']:
        curr = meta['videos'][folder]['objects'][d]
        for f in curr['frames']:
            if f in truth_counter_dict:
                if curr['category'] in truth_counter_dict[f]:
                    truth_counter_dict[f][curr['category']] += 1
                else:
                    truth_counter_dict[f][curr['category']] = 1
            else:
                truth_counter_dict[f] = {curr['category']: 1}
    # for each results objects associated with a frame, count instances and calculate precision and recall
    for r in frame_results:
        frame = r.path.split('.')[0].split('\\')[-1] # name of a frame
        # i = 1
        masks = r.masks  # all the instances seen
        classes = r.boxes.cls  # the annotation of the boxes
        # same form as truth_counter_dict
        # for this frame, it counts the instances of the classes present
        predict_counter_dic = {}
        if masks != None and classes != None:
            for mask, cls in zip(masks, classes):
                if model.names[int(cls)] in predict_counter_dic:
                    predict_counter_dic[model.names[int(cls)]] += 1
                else:
                    predict_counter_dic[model.names[int(cls)]] = 1
                class_name = model.names[int(cls)]
                # print(f"{i} Detected instance of class: {class_name}")
                # i+=1
            # prints how many of each class in the frame
            for key, value in predict_counter_dic.items():
                print(f"{value} {key}")

        # for each frame that has annotations
        if frame in truth_counter_dict:
            # for each class seen in this frame
            for clas in truth_counter_dict[frame]:
                # if the class was predicted and is one we want to search for then calculate precision and recall
                if clas in predict_counter_dic and clas in both_class_Names:
                    if truth_counter_dict[frame][clas] == predict_counter_dic[clas]:
                        class_dic[clas]['AP'].append(1)
                        class_dic[clas]['Recall'].append(1)
                    elif truth_counter_dict[frame][clas] < predict_counter_dic[clas]:
                        class_dic[clas]['AP'].append(
                            truth_counter_dict[frame][clas] / predict_counter_dic[clas])
                        # precision = TruePositives / TruePositives+FalsePositives
                        # so if i had 3 cats in frame, and i predicted 5 cats  -> 3/(3+2) = 0.6
                        class_dic[clas]['Recall'].append(1)
                    elif truth_counter_dict[frame][clas] > predict_counter_dic[clas]:
                        class_dic[clas]['AP'].append(1)
                        class_dic[clas]['Recall'].append(
                            predict_counter_dic[clas] / truth_counter_dict[frame][clas])
                        # if i had 5 cats but predicted 3 it would be 3/(3+2) = 0.75
                        # recall = TruePositives / TruePositives+FalseNegatives
                else:
                    if clas not in classNames and clas not in values_not_in_class_Names:
                        values_not_in_class_Names.append(clas)
                    elif clas in both_class_Names:
                        class_dic[clas]['AP'].append(0)
                        class_dic[clas]['Recall'].append(0)

            for clas in predict_counter_dic:
                if clas not in truth_counter_dict[frame] and clas in both_class_Names:
                    class_dic[clas]['AP'].append(0)
                    class_dic[clas]['Recall'].append(0)

# for each class, calculates mAP and Recall, then calculates for model
total_AP = []
total_Recall = []
for c in class_dic:
    class_dic[c]['AP'] = np.mean(class_dic[c]['AP'])
    class_dic[c]['Recall'] = np.mean(class_dic[c]['AP'])
    print(f"Class {c} mAP: {class_dic[c]['AP']}")
    print(f"Class {c} Recall: {class_dic[c]['Recall']}")
    total_AP.append(class_dic[c]['AP'])
    total_Recall.append(class_dic[c]['Recall'])

print(f"Total AP: {np.mean(total_AP)}")
print(f"Total Recall: {np.mean(total_Recall)}")

with open('measures.txt', 'a') as file:
    for c in class_dic:
        file.write(f"Class {c} mAP: {class_dic[c]['AP']}\n")
        file.write(f"Class {c} Recall: {class_dic[c]['Recall']}\n")
    file.write(f"\nTotal AP: {np.mean(total_AP)}\n")
    file.write(f"Total Recall: {np.mean(total_Recall)}\n")