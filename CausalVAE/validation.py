import torch
from tqdm import tqdm
import sys
sys.path.append(".")
from dag_utils import _h_A
from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_set, model, criterion, optimizer, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    WT_dice = AverageMeter()
    TC_dice = AverageMeter()
    ET_dice = AverageMeter()

    validation_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                    batch_size=opt["validation_batch_size"], 
                                                    shuffle=False,
                                                    pin_memory=True)
    
    val_process = tqdm(validation_loader)
    for i, (inputs, targets, label_distr) in enumerate(val_process):
        # debug
        if targets is None:
            print("Warning: No targets loaded for batch index", i)
        else:
            print(f"Batch {i} - targets size:", targets.size())
            
        if i > 0:
            val_process.set_description("Epoch:%d;Loss:%.4f; dice-WT:%.4f, TC:%.4f, ET:%.4f, lr: %.6f" % (
                                        epoch, losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(),
                                        ET_dice.avg.item(), optimizer.param_groups[0]['lr']))
        if opt["cuda_devices"] is not None:
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
            

        with torch.no_grad():   # increases computation speed during validation
            if opt["CausalVAE_enable"]:
                print("starting val no grad")
                outputs, nelbo, kl, summaries, rec_image, _ = model(inputs, targets, label_distr)
                
                # dag_param adjusted for parallel computing
                if isinstance(model, torch.nn.DataParallel):
                    dag_param = model.module.causal_vae.dag.A
                else:
                    dag_param = model.causal_vae.dag.A
                    
                h_a = _h_A(dag_param, dag_param.size()[0]) # dag acyclicity penalty
                
                L_loss = nelbo + 3*h_a + 0.5*h_a*h_a  # Loss penality
                
                #final loss
                loss = criterion(outputs, targets, inputs, rec_image, L_loss)
                
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        acc = calculate_accuracy(outputs.cpu(), targets.cpu())

        losses.update(loss.cpu(), inputs.size(0))
        WT_dice.update(acc["dice_wt"], inputs.size(0))
        TC_dice.update(acc["dice_tc"], inputs.size(0))
        ET_dice.update(acc["dice_et"], inputs.size(0))

    logger.log(phase="val", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'wt-dice': format(WT_dice.avg.item(), '.4f'),
        'tc-dice': format(TC_dice.avg.item(), '.4f'),
        'et-dice': format(ET_dice.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    return losses.avg, WT_dice.avg, TC_dice.avg, ET_dice.avg