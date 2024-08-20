import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from fcos_core.structures.bounding_box import BoxList

import math
import numpy.random as npr

import cv2
from PIL import Image
import numpy as np

class PascalVOCDataset_V200712(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, is_sample=False):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.is_sample = is_sample
    
        # VOC2007
        self._annopath_voc2007 = os.path.join(self.root, "VOC2007", "Annotations", "%s.xml")
        self._imgpath_voc2007 = os.path.join(self.root, "VOC2007", "JPEGImages", "%s.jpg")
        self._imgsetpath_voc2007 = os.path.join(self.root, "VOC2007", "ImageSets", "Main", "%s.txt")
        with open(self._imgsetpath_voc2007 % self.image_set) as f:
            self.ids_voc2007 = f.readlines()
        self.ids_voc2007 = [x.strip("\n") for x in self.ids_voc2007]
        self.len_ids_voc2007 = len(self.ids_voc2007)
        #self.id_to_img_map = {k: v for k, v in enumerate(self.ids_voc2007)}

        self._annopath_voc2012 = os.path.join(self.root, "VOC2012", "Annotations", "%s.xml")
        self._imgpath_voc2012 = os.path.join(self.root, "VOC2012", "JPEGImages", "%s.jpg")
        self._imgsetpath_voc2012 = os.path.join(self.root, "VOC2012", "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath_voc2012 % self.image_set) as f:
            self.ids_voc2012 = f.readlines()
        self.ids_voc2012 = [x.strip("\n") for x in self.ids_voc2012]
        self.len_ids_voc2012 = len(self.ids_voc2012)
        #self.id_to_img_map = {k: v for k, v in enumerate(self.ids_voc2012)}

        cls = PascalVOCDataset_V200712.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        if index < self.len_ids_voc2007:
            img_id = self.ids_voc2007[index]
            _imgpath = self._imgpath_voc2007
        else:
            index_12 = index - self.len_ids_voc2007
            img_id = self.ids_voc2012[index_12]
            _imgpath = self._imgpath_voc2012

        img = Image.open(_imgpath % img_id).convert("RGB")
        
        if self.is_sample:
            anno = self.get_groundtruth(index)
            boxes = anno["boxes"]
            classes = anno["labels"]
            diffs = anno["difficult"]

            scale_p = npr.randint(0,10) / 10.0
            if scale_p < 0.4:
                scale_p = 1.0
            else:
                scale_p = 0.0
            scale_mark = npr.randint(0,2)*2 - 1
            scale_mark = math.pow(npr.randint(2,6), scale_mark * scale_p)

            im_w = img.size[0] * scale_mark
            im_h = img.size[1] * scale_mark
            im_cx = im_w / 2
            im_cy = im_h / 2

            crop_im_ex = img.size[0] / 2 - im_cx
            crop_im_ey = img.size[1] / 2 - im_cy

            rl_num_boxes = 0
            bboxes_s = []
            gt_classes_s = []
            gt_diff_s = []
            
            if scale_p > 0:
                for idx_b in range(len(boxes)):
                    bb = boxes[idx_b] * 1
                    clsl = classes[idx_b]
                    dffl = diffs[idx_b]

                    if float(bb[2]) - float(bb[0]) > 0 and float(bb[3]) - float(bb[1]) > 0:
                        #bb = (bb.float() * scale_mark).int()

                        bb[0] = int(bb[0] * scale_mark + crop_im_ex)
                        bb[1] = int(bb[1] * scale_mark + crop_im_ey)
                        bb[2] = int(bb[2] * scale_mark + crop_im_ex)
                        bb[3] = int(bb[3] * scale_mark + crop_im_ey)

                        if float(bb[3]) - float(bb[1]) < 16 or float(bb[2]) - float(bb[0]) < 16 or \
                            float(bb[0]) < 0 or float(bb[1]) < 0 or bb[2] > img.size[0] or bb[3] > img.size[1]:
                            continue
                        rl_num_boxes += 1
                        bboxes_s.append(bb)
                        gt_classes_s.append(clsl)
                        gt_diff_s.append(dffl)
            
            if rl_num_boxes > 1:
                center = (img.size[0] // 2, img.size[1] // 2)
                scale_mat = cv2.getRotationMatrix2D(center, 0, scale_mark)
                imgv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                padding = np.mean(imgv, axis=(0, 1))

                imgv = cv2.warpAffine(imgv, scale_mat, (img.size[0], img.size[1]),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=padding)
                img = Image.fromarray(cv2.cvtColor(imgv,cv2.COLOR_BGR2RGB))

                boxes = torch.stack(bboxes_s)
                classes = torch.stack(gt_classes_s)
                diffs = torch.stack(gt_diff_s)
                '''
                if index < self.len_ids_voc2007:
                    for idx_b in range(len(bboxes_s)):
                        bb = bboxes_s[idx_b]
                        cv2.rectangle(imgv, (int(bb[0]), int(bb[1])),(int(bb[2]),int(bb[3])), (0, 255-idx_b*10, 0+10*idx_b), 3)
                    cv2.imwrite('./images_data/'+str(index)+'.jpg', imgv)
                '''
            
            height, width = anno["im_info"]
            target = BoxList(boxes, (width, height), mode="xyxy")
            target.add_field("labels", classes)
            target.add_field("difficult", diffs)
        else:
            target = self.get_groundtruth(index)

        #target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids_voc2007) + len(self.ids_voc2012)

    def get_groundtruth(self, index):
        #img_id = self.ids[index]
        if index < self.len_ids_voc2007:
            img_id = self.ids_voc2007[index]
            anno = ET.parse(self._annopath_voc2007 % img_id).getroot()
        else:
            index_12 = index - self.len_ids_voc2007
            img_id = self.ids_voc2012[index_12]
            anno = ET.parse(self._annopath_voc2012 % img_id).getroot()
            
        #anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)
        
        if self.is_sample:
            return anno
        else:
            height, width = anno["im_info"]
            target = BoxList(anno["boxes"], (width, height), mode="xyxy")
            target.add_field("labels", anno["labels"])
            target.add_field("difficult", anno["difficult"])
            return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            #if 'background' in name:
            #print('------00', name, self.class_to_ind[name])
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text, 
                bb.find("ymin").text, 
                bb.find("xmax").text, 
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        if index < self.len_ids_voc2007:
            img_id = self.ids_voc2007[index]
            _annopath = self._annopath_voc2007
        else:
            index_12 = index - self.len_ids_voc2007
            img_id = self.ids_voc2012[index_12]
            _annopath = self._annopath_voc2012

        anno = ET.parse(_annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset_V200712.CLASSES[class_id]
