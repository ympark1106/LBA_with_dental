import os
import time
import argparse
import warnings
warnings.filterwarnings(action='ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import torch

from loader.panorama_coco import CocoDataset, collate_fn
from models.maskrcnn import dental

def main(exp_name, args):
    is_save = args.save
    if is_save:
        save_path = os.path.join(args.save_dir, '{}'.format(exp_name)).replace(" ", "_")
        save_path = save_path.replace(":", "-")
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        f = open(os.path.join(save_path, "info.txt"), 'w')
        f.write(str(args))

        f.close()
        
    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device("cpu")
    
    root = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/images" 
    json_path = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/annotations/instances.json"

    
    train_dataset = CocoDataset(root=root, json=json_path, train=True)
    print("train_dataset size : {}".format(train_dataset.__len__()))
    # exit()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False,num_workers=4, shuffle=True, collate_fn=collate_fn)
    
    num_classes = len(train_dataset.class_cate)
    model = dental(num_classes=num_classes).to(args.device)
    parameters = [p for p in model.parameters() if p.requires_grad]


    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    
    print_freq = 5
    for epoch in range(args.epochs):
        
        loss_sum = 0
 
        
        model.train()
        for i, data in enumerate(train_loader):
            
            
            lr_scheduler = None
  
            labeled_image, labeled_target = data
            
    
            labeled_image = list(image.to(args.device) for image in labeled_image)
            labeled_target = [{k: v.to(args.device) for k, v in t.items()} for t in labeled_target]


            optimizer.zero_grad()
            
            losses = model(labeled_image, labeled_target)
            # print(losses)
            loss = sum(loss for loss in losses.values())
            
            loss.backward()
            optimizer.step()
            
  
            loss_sum += loss.item()
            loss_avg = loss_sum / (i+1)
            
            if i % print_freq == 0:
                print(loss_avg)
            
            
            if lr_scheduler is not None:
                lr_scheduler.step()
                
        scheduler.step()
        print("-"*80)
        
        if args.save:
            torch.save(model.state_dict(), os.path.join(save_path, "epoch{}.pth".format(str(epoch))))    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument("--scheduler", type=str, default="StepLR", help="learning rate scheduler")
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    parser.add_argument("--pretrained", type=bool, default=True, help="small model pretrained by imagenet")
    
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/Datasets", help="dataset name")
    parser.add_argument("--dataset-name", type=str, default="coco", help="dataset name")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument(
        "--lr-steps",
        default=[20, 30],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="checkpoints")
 
    args = parser.parse_args()
    exp_name = time.strftime('%c', time.localtime(time.time()))

    main(exp_name, args)