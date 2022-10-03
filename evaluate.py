import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys
sys.path.append('./')
import coco_names
import image_utils
from engine import evaluate

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')

    # parser.add_argument('--model_path', type=str, default='./result/model_19.pth', help='model path')
    parser.add_argument('--evaluate_dir', type=str, default='./test.jpg', help='image path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    input = []
    if args.dataset == 'coco':
        num_classes = 91
        names = coco_names.names
        
    # Model creating
    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=True)
    device = torch.device("cuda")


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.b,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    evaluate(model, data_loader_test, device=device)

    # 加载模型
    # save = torch.load(args.model_path)
    # model.load_state_dict(save['model'])

    # 加载图片，转换颜色归一化
    src_img = cv2.imread(args.image_path)
    img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().cuda()
    input.append(img_tensor)

    # inference
    out = model(input)

    # 结果展示
    src_img = image_utils.draw_picture(src_img, out, names, args.score)
    cv2.imshow('result',src_img)   
    cv2.waitKey()
    cv2.destroyAllWindows()

    # cv2.imwrite('assets/11.jpg',img)
    

if __name__ == "__main__":
    main()
