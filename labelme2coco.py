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

    def to_coco(self, labelme_path):
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

        
        #for json_path in (json_path_list):
        obj = self.read_jsonfile(labelme_path)
        self.images.append(self._image(obj, labelme_path))
        shapes = obj['shapes']
            
            # check number of people in the image
        num_person = 0
        for shape in shapes:
            if shape['group_id'] > num_person:
                num_person = shape['group_id']
        
        # do annotation for each person
        for person in range(num_person+1):
            person_annotation = []
            keypoints =  [None] * 18
            
            for shape in shapes:
                if shape['group_id'] != person:
                    # group id also starts with 0, i.e. person 0, person 1, ...
                    continue
                # record body parts for person, part_index max = 17
                part_index = int(shape['label']) 
                keypoints[part_index] = shape['points'][0]

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
            self.ann_id += 1
            annotation['id'] = self.ann_id
            annotation['image_id'] = self.img_id
            annotation['category_id'] = 1
            annotation['iscrowd'] = 0
            
            annotation['num_keypoints'] = num_keypoints
            annotation['keypoints'] = person_annotation
            
            # add person annotation to image annotation
            self.annotations.append(annotation)
                
            # next image
            # self.img_id += 1
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


    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    labelme_path = "./frame_10001.json"
    test = Lableme2CoCo()
    train_instance = test.to_coco(labelme_path)
    
    json.dump(train_instance, open('./keypoints_train.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        
        