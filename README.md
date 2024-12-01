# AnimalStream
Machine Learning - Computer Vision Project: Image Classification of Web Camera Streams
Created by Lauren Lyons.
The github repo is https://github.com/snoylnerual/AnimalStream.

AnimalStream Directory Format

AnimalStream<br>
|<br>
| - data<br>
|&emsp;&emsp;|-train/train<br>
|&emsp;&emsp;|&emsp;&emsp;| - Annotations<br>
|&emsp;&emsp;|&emsp;&emsp;| - JPEGImages<br>
|&emsp;&emsp;| - valid/valid<br>
|&emsp;&emsp;|&emsp;&emsp;| - Annotations<br>
|&emsp;&emsp;|&emsp;&emsp;| - JPEGImages<br>
|<br>
| - runs<br>
|&emsp;&emsp;| - segment<br>
|<br>
| - yolo-Weights<br>

This is how this repo is laid out. In the AnimalStream folder, there is the main.py, which runs the model and 
all associated code. There is also measures.txt. This holds the output of the program, including the mAP and Recall
of all classes that we consider valid. Valid classes are the ones seen in both the COCO dataset and Youtube VIS dataset.

In the data folder, there are train and valid splits. In both of these folder, they hold the Annotations and JPEGImages
folders. There are also meta.json and meta_instance.json files in these folders that hold the information of the videos,
class instances, and frames that said class instances are present in. Inside the Annotation folder, there are the
annotations of each video inside. In JPEGImages, there are multiple videos, where each one is a folder. In that folder,
there are all the frames of a video as jpeg files.

In the runs folder, there is a segment folder that holds all the information for the different runs, like the specific
weights and arguments used for each model. 

In the yolo-Weights folder, there are the weights of the yolo models. Currently, we are using the yolov8s-seg model.
There are multiple different models, versions, and tasks that can be chosen. The models range in size, going from
nano (n) to extra-large (x). There are also versions from number 1 to 11 currently. Then, some of the tasks that can
be chosen are object detection, instance segmentation, pose/keypoints, oriented detection, and classification. More
information can be found at https://docs.ultralytics.com/. 

------------------------------------------------------------------------------------------------------------------------

main.py is the main driver of this project. There are no specific inputs that you change outside of the code, but there
are specific items that can be changed within the code. The type of yolo model can be changed, the confidence, and the
IoU value can all be changed for the results. 

The data input is from the YoutubeVIS dataset. The link to it is here: https://youtube-vos.org/dataset/vis/. There
are a few steps that you must go through in order to get the dataset downloaded, like making an account for CodaLab
and joining the competition. 

The output is the instances of multiple objects in each frame. Like, if there are 4 cats in a video frame, it should
report that there were 4 cats in the frame. Where this comes into relevance for our project, is that we want to be
able to see which animals are present in a video webcam frame, and report that. Ideally, we want to report how many
of all the types of animals that passed by the camera for multiple reasons. It is first a fun data point for outside
customers to see for animal webcams. If there are certain animals that can pass by, it is fun to see that it is possible
to see those animals, and it would encourage watchers to attend for the rare animals sightings. We can also track the 
activity of these animals, and the customers can view this as well. All of these promote engagement and surveillance of 
these animals. If they were to change their habits and patterns, then it could be readily available to see, and the 
animals could receive care sooner. We can also test the performance of these models on this kind of data when we run it.
There are other reasons that we would want to run this model on this sort of data and complete these certain tasks!

The evaluation metrics of the outputs that we use are mAP and Recall. 
- precision = TruePositives / TruePositives+FalsePositives
- recall = TruePositives / TruePositives+FalseNegatives

For each class, we individually evaluate the precision and recall for every single frame that the class is present in,
whether it be in the truth or prediction. For precision, if there are 4 cats in the truth, but we predicted there was
5 cats, our evaluation would be 'precision = 4 / (4+1) = 0.8'. That would be for a single frame. For recall, if there
are 4 cats in the truth, but we predicted 3 cats, the evaluation would be 'recall = 3 / (3+1) = 0.75'. That would be for
a single frame. To get the final mAP and Recall for each class, we just take the average of all precision and recall
for all the different frames that we used. The total mAP and Recall is the average of all the different classes' mAPs 
and Recalls.

The output of measures.txt results in a total Recall and total mAP of:

- Total AP: 0.3366
- Total Recall: 0.3366

------------------------------------------------------------------------------------------------------------------------

We YOLOv8 in this program. We use the small version that is trained under the instance segmentation task. Our usage was
developed by Ultralytics.

The confidence score in YOLOv8 represents the model's certainty about its predictions. It combines two types of 
confidence: box confidence and class confidence. Box confidence is the probability that a bounding box contains
an object. The class confidence is the likelihood that a detected object belongs to a particular class. Then the final
confidence score is calculated by multiplying these two confidences, which will result in a range from 0 to 1.
Higher scores indicating greater certainty in the detection.

The IoU score is Intersection over Union. It measures the overlap between two bounding boxes: the predicted bounding 
box (output by the model) and the ground truth bounding box (from the dataset). It helps assess how accurately the 
model predicts object locations It is calculated by IoU is calculated as follows: 
IoU = Area of Intersection / Area of Union. The output of IoU results in a range from 0 to 1. Ultralytics says that 
IoU > 0.5 is considered decent, IoU > 0.7 is considered good, and IoU > 0.9 is considered almost perfect. 