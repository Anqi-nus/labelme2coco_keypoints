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
|   ... mydata (directory to store generated COCO format .json and images)    
|       ... annotations  
|           ... person_keypoints_train2017.json  
|           ... person_keypoints_val2017.json  
|       ... images  
|           ... train2017  
|               ...  
|               ...  
|           ... val2017  
|               ...  
|               ...  
</pre>

reference https://github.com/m5823779/labelme2coco_keypoint/blob/master/labelme2coco.py  
