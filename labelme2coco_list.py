import os
import json
import numpy as np
import glob
import shutil
from tqdm import tqdm
from labelme import utils
from sklearn.model_selection import train_test_split
np.random.seed(41)

classname_to_id = {"person": 1}
labels = ["nose", #1
                "left_eye", #2
                "right_eye", #3
                "left_ear", #4
                "right_ear", #5
                "left_shoulder", #6
                "right_shoulder", #7
                "left_elbow", #8
                "right_elbow", #9
                "left_wrist", #10
                "right_wrist", #11
                "left_hip", #12
                "right_hip", #13
                "left_knee", #14
                "right_knee", #15
                "left_ankle", #16
                "right_ankle"] #17

class Lableme2CoCo:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def to_coco(self, json_path_list):
        self._init_categories()
        instance = {}
        instance['info'] = {'description': 'Pose Estimation Dataset', 
                            'version': 1.0, 
                            'year': 2021,
                            'contributer': "conex",
                            'date_created': "2021/02/08" }
        instance['license'] = ['Conex']
        instance['images'] = self.images
        instance['categories'] = self.categories
        
        for json_path in (json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
                
            # check number of people in the image
            num_person = 0
            isIndividual = False
            for shape in shapes:
                if shape['group_id'] == None:
                    isIndividual = True
                    continue
                if shape['group_id'] > num_person:
                    num_person = shape['group_id']
            
            print("Start annotate for img: ", json_path, "There are", num_person + 1, "people in total")
            # do annotation for each person
            for person in range(num_person + 1):
                print("Person", person + 1, "...")
                # start with person = 0, create annotation dict for each person
                person_annotation = []
                keypoints =  [None] * 18
                
                for shape in shapes:
                    # iterate through keypoints, add to dict if belongs to person
                    if shape['group_id'] != person and isIndividual == False:
                        continue
                    # get the body part this keypoint represents
                    part_index = int(shape['label']) 
                    # store the keypoint data to keypoints[] at its respective index
                    keypoints[part_index] = shape['points'][0]

                # edit the keypoint data to fit COCO annotation format
                num_keypoints = 0
                for keypoint_i in range(1, 18): 
                    # store keypoint for person in annotation
                    if keypoints[keypoint_i] == None:
                        person_annotation.extend([0, 0, 0])
                    else:
                        person_annotation.extend([keypoints[keypoint_i][0], keypoints[keypoint_i][1], 2])
                        num_keypoints += 1                    
                    
                # annotate all other information for this person
                annotation = {}
                annotation['id'] = self.ann_id
                annotation['image_id'] = self.img_id
                annotation['category_id'] = 1
                annotation['iscrowd'] = 0
                annotation['num_keypoints'] = num_keypoints
                annotation['keypoints'] = person_annotation
                # add person annotation to image annotation
                print("Annotated data: ", annotation)
                self.annotations.append(annotation) 
                self.ann_id += 1
                
            # next image
            self.img_id += 1
        
        # store to output .json instance
        instance['annotations'] = self.annotations
        return instance
        
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['supercategory'] = k
            category['id'] = v
            category['name'] = k
            category['keypoints'] = ["nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"]
            category['skeleton'] = [
                [
                    16,
                    14
                ],
                [
                    14,
                    12
                ],
                [
                    17,
                    15
                ],
                [
                    15,
                    13
                ],
                [
                    12,
                    13
                ],
                [
                    6,
                    12
                ],
                [
                    7,
                    13
                ],
                [
                    6,
                    7
                ],
                [
                    6,
                    8
                ],
                [
                    7,
                    9
                ],
                [
                    8,
                    10
                ],
                [
                    9,
                    11
                ],
                [
                    2,
                    3
                ],
                [
                    1,
                    2
                ],
                [
                    1,
                    3
                ],
                [
                    2,
                    4
                ],
                [
                    3,
                    5
                ],
                [
                    4,
                    6
                ],
                [
                    5,
                    7
                ]
            ]
            self.categories.append(category)

    def _image(self, obj, path):
        image = {}
        img_x = utils.img_b64_to_arr(obj['imageData'])
        image["height"] = img_x.shape[0]
        image["width"] = img_x.shape[1]
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image


    # read json file, return json object
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

if __name__ == '__main__':
    print("---------------------------------")
    image_path = "./annotated/"
    saved_coco_path = "./annotated/"
    
    json_list_path = glob.glob(image_path + "/*.json")
    train_path, val_path = train_test_split(json_list_path, test_size=0.2)
    print(train_path)
    print(val_path)
    
    train = Lableme2CoCo()
    train_instance = train.to_coco(train_path)
    
    val = Lableme2CoCo()
    val_instance = val.to_coco(val_path)
    
    # save to file path
    train_save_path = "./converted/mydata/annotations/person_keypoints_train2017.json"
    val_save_path = "./converted/mydata/annotations/person_keypoints_val2017.json"
    json.dump(train_instance, open(train_save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(val_instance, open(val_save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        
        