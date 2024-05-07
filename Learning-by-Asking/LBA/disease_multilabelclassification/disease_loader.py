import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from PIL import Image

from torchvision import transforms

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

import albumentations as A

import numpy as np

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def label_base_sampling(img_list, img_labels):

    
    category_list = np.unique(img_labels)
    

    category_index_dict = {cate:[] for cate in category_list}
    # usage_list = [100, 47, 100, 100, 36, 100, 100, 100, 100, 100]
    # usage_list = [200, 47, 200, 200, 36, 215, 317, 163, 132, 206]
    
    # usage_list = [313, 3684, 1341, 84, 547, 979, 628,  447, 2280]
    usage_list = [313, 500, 500, 84, 500, 500, 500, 447, 500]

    for idx, lb in enumerate(img_labels):
        category_index_dict[lb].append(idx)
    
    new_indice = []
    for usage, cate_i in zip(usage_list, category_index_dict.values()):
        new_indice += cate_i[:usage]

    new_img_list = [img_path for idx, img_path in enumerate(img_list) if idx in new_indice]
    new_label_list = [img_label for idx, img_label in enumerate(img_labels) if idx in new_indice]

    
    return new_img_list, new_label_list



class DentalDisease(Dataset):
    def __init__(self, train, root):
        self.root = os.path.join(root, "cropped_set")
        self.train = train
        self.resize = (224, 224)

        label_type = "Multilabel"
        # label_type = "Multiclass"
        
        total_img_list, label_list = self.read_imgfolder(self.root)
        
        img_list = []
        img_labels = []
        
        
        assert len(total_img_list) == len(label_list), "img list and label lenght is not same"
        
        if label_type == "Multiclass":
        
            keep = [idx for idx, (_, lb) in enumerate(label_list) if len(lb) == 1]
            
            img_list   = [img_path for idx, img_path in enumerate(total_img_list) if idx in keep]
            img_labels = [lb[0]-33 for idx, (_, lb) in enumerate(label_list) if idx in keep]
        
        
        elif label_type == "Multilabel":
            
            img_list   = total_img_list
            img_labels = [lb for _, _, lb in label_list]
            ori_img_indice = [im_idx for im_idx, _, _ in label_list]
            

            d_count_list = [0, 0, 0, 0, 0, 0, 0, 0]
            img_labels = []
            for idx, (_, _, lb) in enumerate(label_list):
                temp = [0, 0, 0, 0, 0, 0, 0, 0]

                lb = list(set(lb))
           
                for i in lb:
                    
                    # if i>=34 and i<36:
                    #     temp[0] = 1
                    #     d_count_list[0] +=1
                    
                    if i>=36 and i<48:
                        temp[0] = 1
                        d_count_list[0] +=1
                        
                    elif i>=49 and i<51:
                        temp[1] = 1
                        d_count_list[1] +=1
                        
                    # elif i>=52 and i<54:
                    #     temp[3] = 1
                    #     d_count_list[3] +=1
                        
                    # elif i>=54 and i<57:
                    #     temp[4] = 1
                    #     d_count_list[4] +=1
                    
                    elif i>=57 and i<58:
                        temp[2] = 1
                        d_count_list[2] +=1
                        
                    # elif i>=60 and i<64:
                    #     temp[6] = 1
                    #     d_count_list[6] +=1
                        
                    elif i>=64 and i<65:
                        temp[3] = 1
                        d_count_list[3] +=1
                    
                    elif i==67:
                        temp[4] = 1
                        d_count_list[4] +=1
                    
                    # elif i==68 or i==69:
                    #     temp[9] = 1
                    #     d_count_list[9] +=1
                    
                    elif i==70:
                        temp[5] = 1
                        d_count_list[5] +=1
                        
                    elif i==71:
                        temp[6] = 1
                        d_count_list[6] +=1
                    
                    elif i==72:
                        temp[7] = 1
                        d_count_list[7] +=1
                    

                    

                    # if i == 37:
                    #     print(idx)
                    
                img_labels.append(temp)
        
        print(d_count_list)
        # exit()
        train_indice = []
        test_indice  = []
        
        test_inst_const = 50
        test_inst_count = np.array(len(d_count_list) * [0])
        train_inst_count = np.array(len(d_count_list) * [0])
        
        test_raw_img_indice = []
        
        test_set_collect = True
        for idx, (ori_img_idx, i_img_list, i_img_lb) in enumerate(zip(ori_img_indice, img_list, img_labels)):
            
            cate_label = np.where(np.array(i_img_lb) == 1)[0].tolist()
            
            for cate in cate_label:
                if test_inst_count[cate] >= test_inst_const:
                    test_set_collect = False
                    break
                    
            if test_set_collect:
                test_inst_count = test_inst_count + np.array(i_img_lb)
                test_indice.append(idx)
                test_raw_img_indice.append(ori_img_idx) 
            elif (test_set_collect == False) and (ori_img_idx not in test_raw_img_indice):
                train_inst_count = train_inst_count + np.array(i_img_lb)
                train_indice.append(idx)
            
            test_raw_img_indice = list(set(test_raw_img_indice))
            test_set_collect = True
        
        # print(test_inst_count)
        # print(train_inst_count)
        # exit()
        normal_img_list, normal_labels = self.read_imgfolder(self.root, mode="normal", label_type=label_type)
                 
        # if label_type == "Multiclass":
        #     img_list, img_labels = label_base_sampling(img_list, img_labels)
        
        

        X_train = np.array(img_list)[train_indice]
        y_train = np.array(img_labels)[train_indice]
        
        
        # X_train  = X_train.tolist() + normal_img_list[:800]
        # y_train = y_train.tolist() + normal_labels[:800]

        

        X_test = np.array(img_list)[test_indice]
        y_test = np.array(img_labels)[test_indice]
        
        if label_type == "Multilabel":
            d_count_list = [0, 0, 0, 0, 0, 0, 0, 0]
            im_path_cat_list = [[],[],[],[],[],[],[],[]]
            
            im_mtlb_cat_list = [[],[],[],[],[],[],[],[]]
            
            for im_path, multi_lb in zip(X_train, y_train):                
                for idx, lb in enumerate(multi_lb):
                    if lb == 1:
                        d_count_list[idx] +=1
                        im_path_cat_list[idx].append(im_path)
                        im_mtlb_cat_list[idx].append(multi_lb)
                    

            d_len = 400
            new_img_list = [] 
            new_img_labels = []
            
            for idx, (d_cout, im_path_list, im_mtlb_list) in enumerate(zip(d_count_list, im_path_cat_list, im_mtlb_cat_list)):

                upsamp_im_list = im_path_list * (round(d_len/d_cout)+1)
                upsamp_im_list = upsamp_im_list[:d_len]
                
                new_img_list += upsamp_im_list
                
                upsamp_lb_list = []
                for _ in range((round(d_len/d_cout)+1)):
                    upsamp_lb_list += im_mtlb_list
                    
                upsamp_lb_list = upsamp_lb_list[:d_len]
                new_img_labels += upsamp_lb_list


            X_train = new_img_list
            y_train = new_img_labels
            
            
            # normal_img_list, normal_labels = self.read_imgfolder(self.root, mode="normal", label_type=label_type)
            

        if train:
            self.img_list = X_train
            self.img_labels = y_train
            # print(np.unique(y_train, return_counts=True))
        
        else:
            self.img_list = X_test
            self.img_labels = y_test
      
        
        self.clahe = A.Compose([
                        A.CLAHE(p=1.0, clip_limit=(1, 4), tile_grid_size=(4, 4))
                    ])


        if self.train:
            self.transform = transforms.Compose([
                        transforms.Resize(size=(int(self.resize[0]), int(self.resize[1]))),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)),
                        
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225)),])

        else:
            self.transform = transforms.Compose([
                        transforms.Resize(size=(int(self.resize[0]), int(self.resize[1]))),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225)),])


    def read_imgfolder(self, path, mode="disease", label_type=None):
        total_img_list = []
        
        if mode == "disease":
                
            f = open("/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_set/labels.txt", 'r')
            lines = f.readlines()
            labels = []
            for ln in lines:

                # img_id, label = ln[:-2].split('/[')
                raw_img_id, img_id, label = ln.split('/')
                label = eval(label)
             
                labels.append([int(raw_img_id), int(img_id), label])
                # labels.append([int(img_id), list(map(int, label.split(',')))])
                
                img_path = os.path.join(path, "img_disease", img_id+".jpg")
                total_img_list.append(img_path)
                
            
                
        
        elif mode == "normal":
            for f_name in os.listdir(os.path.join(path, "img_normal")):
                img_path = os.path.join(path, "img_normal", f_name)
                total_img_list.append(img_path)
            
            if label_type == "Multilabel":
                
                labels = []
                for _ in range(len(total_img_list)):
                    labels.append([0, 0, 0, 0, 0, 0, 0, 0])
    
            else:
                labels = len(total_img_list) * [0]


        return total_img_list, labels
    

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx, vis=False):
        img_path = self.img_list[idx]
        
        path = os.path.join(self.root, img_path)
        
        img = Image.open(path).convert('RGB')
        img = Image.fromarray(self.clahe(image = np.array(img))['image'])
        
        img = expand2square(img, (0, 0, 0))
        

    
        img = self.transform(img)
            
        target = torch.as_tensor(self.img_labels[idx])

        if vis:
            return img, target, path
            
        else:
            return img, target



if __name__ == "__main__":
    s = DentalDisease(train=True, root="/home/gpu/Workspace/youmin/Learning-by-Asking/LBA")
    s.__getitem__(1)