# labelme2coco_keypoints
Convert labelme format keypoints .json to MSCOCO format keypoints for human detection  

Install labelme before running the script.  
Installation Steps For Windows:  
-	Install Anaconda using this link: https://www.anaconda.com/products/individual#windows
-	Open Anaconda, run the commands in Anaconda Prompt:  
	`conda create --name=labelme python=3.6`  
	`conda activate labelme`  
	`conda install labelme -c conda-forge`  



<pre>
Directory  
...  
|   ... labelme2coco_list.py   
|   ... annotated (directory to store raw annotated labelme .json files)   
|   ... converted_folder (directory to store generated COCO format .json and images)    
|       ... annotations  
|           ... person_keypoints_train.json  
|           ... person_keypoints_val.json  
|       ... images  
|           ... train  
|               ...  
|               ...  
|           ... val  
|               ...  
|               ...  
</pre>

reference https://github.com/m5823779/labelme2coco_keypoint/blob/master/labelme2coco.py  
