import torch
import torchvision
import torchvision.transforms as transform

import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import  transforms
import pickle
from torchvision.utils import save_image



with open('gt_id_conf.pickle', mode='rb') as fi:
	gt_id = pickle.load(fi)


with open('gt_bbox_conf.pickle', mode='rb') as fi:
	gt_bbox = pickle.load(fi)

to_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor()
	])


class TTUDataset():
	
	def __init__(self,transform,train=True):
		self.transform = transform
		self.number_train=["MOT16-02","MOT16-04","MOT16-05","MOT16-09","MOT16-10","MOT16-11","MOT16-13"]
		if train==True:
			
			with open('gt_id.pickle', mode='rb') as fi:
				self.gt_id = pickle.load(fi)
			with open('gt_bbox.pickle', mode='rb') as fi:
				self.gt_bbox = pickle.load(fi)
		
	def get_data(self,batch_size,image_size):
		number=random.choice(self.number_train) #使用データセットフォルダをランダムに選択
		id_num_list=random.sample(list(gt_id[number].keys()),batch_size)
		print(number,id_num_list)
		previous_image_data=[]
		current_image_data=[]
		for id_num in id_num_list: #(sequence,batch_size,channel,w,h)の入力を作る
			frame_list=gt_id[number][id_num][0]
			#current_frame=random.randint(frame_list[5], frame_list[-1])
			current_frame_id=random.randint(5, len(frame_list)-1)
			print(len(frame_list),"len_frame_list")
			print(current_frame_id,"current_frame_id")
			previous_frame=frame_list[current_frame_id-5:current_frame_id]
			current_frame=frame_list[current_frame_id]
			print(previous_frame,"previous_frame")
			previous_image_list=[]

			bounding_box=gt_bbox[number][id_num][current_frame]
			img_path="MOT16/train/"+number+"/img1/{0:06d}.jpg".format(int(current_frame))
			with open(img_path, 'rb') as f:
				image = Image.open(f)
				image = image.convert('RGB')
			image_cut=image.crop((bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]))
			image_tensor= self.transform(image_cut)
			current_image_data.append(image_tensor)


			for f in previous_frame:
				print(f,"f")
				bounding_box=gt_bbox[number][id_num][f]
				print(bounding_box,"bounding_box")
				img_path="MOT16/train/"+number+"/img1/{0:06d}.jpg".format(int(f))
				with open(img_path, 'rb') as f:
					image = Image.open(f)
					image = image.convert('RGB')
				image_cut=image.crop((bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]))
				#image=cv2.imread(img_path)
				#image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
				#image_cut=image[bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2],:]
				#mage_cut = Image.fromarray(np.uint8(image_cut))
				image_tensor= self.transform(image_cut)
				previous_image_list.append(image_tensor)
			print(len(previous_image_list),"len_previous_list")
			previous_image_list = torch.stack(previous_image_list,dim=0)
			previous_image_data.append(previous_image_list)


		current_image_data=torch.stack(current_image_data,dim=0) #(batch_size,C,W,H)
		previous_image_data=torch.stack(previous_image_data,dim=0)
		previous_image_data=torch.transpose(previous_image_data,0,1) #(t,batch_size,C,W.H)の形式にする

		
		return previous_image_data,current_image_data
Dataset=TTUDataset(to_transform)
previous_image_data,current_image_data=Dataset.get_data(batch_size=16,image_size=224)

previous_image_data=torch.transpose(previous_image_data,0,1) #(batch_size,t,C,W.H)
for i in range(previous_image_data.size()[0]):
	save_image(previous_image_data[i],"save_image/sample_previous_input{}.jpg".format(i))

save_image(current_image_data,"save_image/sample_current_input.jpg")



