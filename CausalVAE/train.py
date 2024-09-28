import torch
from tqdm import tqdm
import sys
import numpy as np
from torchvision.utils import save_image
from dag_utils import _h_A
sys.path.append(".")

from utils import AverageMeter, calculate_accuracy, calculate_accuracy_singleLabel, process_middle_slice
from PIL import Image

def train_epoch(epoch, data_set, model, criterion, optimizer, opt, logger, extra_train=False):
    print('train at epoch {}'.format(epoch))
    
    # initial setup
    model.train() # set model to training mode

    # average meter instances to keep track of loss and dice scores across the epoch
    losses = AverageMeter() 
    WT_dice = AverageMeter()
    TC_dice = AverageMeter()
    ET_dice = AverageMeter()

    # data_set.file_open()
    train_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                               batch_size=opt["batch_size"], 
                                               shuffle=True, 
                                               pin_memory=True)
            
            
    training_process = tqdm(train_loader) # progress bar
    for i, (inputs, targets, label_distr) in enumerate(training_process):
        # debug
        if targets is None:
            print("Warning: No targets loaded for batch index", i)
        else:
            print(f"Batch {i} - targets size:", targets.size())
            

            
        if i > 0:
            training_process.set_description("Epoch:%d;Loss:%.4f; dice-WT:%.4f, TC:%.4f, ET:%.4f, lr: %.6f"%(epoch,
                                             losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(),
                                             ET_dice.avg.item(), optimizer.param_groups[0]['lr']))

        if opt["cuda_devices"] is not None:
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
            label_distr = label_distr.type(torch.FloatTensor)
            label_distr = label_distr.cuda()

        # forward pass
        if opt["CausalVAE_enable"]:
            outputs, nelbo, kl, summaries, rec_image, _ = model(inputs, targets, label_distr)

            # dag_param adjusted for parallel computing
            if isinstance(model, torch.nn.DataParallel):
                dag_param = model.module.causal_vae.dag.A
            else:
                dag_param = model.causal_vae.dag.A

            h_a = _h_A(dag_param, dag_param.size()[0]) # dag acyclicity penalty
            L_loss = nelbo + 3*h_a + 0.5*h_a*h_a  # final loss for cvae
            
            loss = criterion(outputs, targets, inputs, rec_image, L_loss)
            print("Gradients of DAG matrix A: ", model.module.causal_vae.dag.A.grad) 
            

            #save input images to compare, whole image and middle slice as png
            inputs_array, inputs_middle_slice = process_middle_slice(inputs)
            np.save('figs_cvae/3D/original_image_{}.npy'.format(epoch), inputs_array)
            inputs_middle_slice.save('figs_cvae/MiddleSlice/input_image_{}.png'.format(epoch))
            print("input images saved..")
            
            #save reconstructed images, whole image and middle slice png
            rec_array, rec_middle_slice = process_middle_slice(rec_image)
            np.save('figs_cvae/3D/original_image_{}.npy'.format(epoch), rec_array)
            rec_middle_slice.save('figs_cvae/MiddleSlice/reconstructed_image_{}.png'.format(epoch))
            print("save rec images fine..")
            
            print('Training info: summaries:{}'.format(summaries))
            print('Dag param:{}'.format(dag_param))
                            
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if opt["flooding"]:
            b = opt["flooding_level"]
            loss = (loss - b).abs() + b  # flooding

        if not opt["seg_dice"]:
            acc = calculate_accuracy(outputs.cpu(), targets.cpu())  # dice_coefficient
        else:
            acc = dict()
            acc["dice_wt"] = torch.tensor(0)
            acc["dice_tc"] = torch.tensor(0)
            acc["dice_et"] = torch.tensor(0)
            singleLabel_acc = calculate_accuracy_singleLabel(outputs.cpu(), targets.cpu())
            acc[opt["seg_dice"]] = singleLabel_acc

        losses.update(loss.cpu(), inputs.size(0))  # batch_avg
        WT_dice.update(acc["dice_wt"], inputs.size(0))
        TC_dice.update(acc["dice_tc"], inputs.size(0))
        ET_dice.update(acc["dice_et"], inputs.size(0))

        # backprop and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # logger
    logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'wt-dice': format(WT_dice.avg.item(), '.4f'),
        'tc-dice': format(TC_dice.avg.item(), '.4f'),
        'et-dice': format(ET_dice.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr'],
        'dag': model.module.causal_vae.dag.A
    })

    if extra_train:
        return losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(), ET_dice.avg.item()

