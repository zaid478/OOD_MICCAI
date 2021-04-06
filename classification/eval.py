#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file function to evaluate a model

If you find any bug or have some suggestion, please, email me.
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from raug.raug.metrics import Metrics, accuracy
from raug.raug.checkpoints import load_model
from raug.raug.utils.classification_metrics import AVGMetrics
from tqdm import tqdm


def metrics_for_eval (model, data_loader, device, loss_fn, topk=2, get_balanced_acc=False, get_auc=False):
    """
        This function returns accuracy, topk accuracy and loss for the evaluation partition

        :param model (nn.Model): the model you' want to evaluate
        :param data_loader (DataLoader): the DataLoader containing the data partition
        :param checkpoint_path(string, optional): string with a checkpoint to load the model. If None, none checkpoint is
        loaded. Default is None.
        :param loss_fn (nn.Loss): the loss function used in the training
        :param get_balanced_acc (bool, optional); if you want to compute the balanced accuracy for the eval partition
        set it as True. The default is False because it may impact in the training phase.
        :param get_auc (bool, optional); if you want to compute the AUC for the eval partition
        set it as True. The default is False because it may impact in the training phase.
        :return: a instance of the classe metrics
    """

    # setting the model to evaluation mode
    model.eval()

    print ("\nEvaluating...")
    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, ncols=100) as t:

        # Setting require_grad=False in order to dimiss the gradient computation in the graph
        with torch.no_grad():

            loss_avg = AVGMetrics()
            acc_avg = AVGMetrics()
            topk_avg = AVGMetrics()

            opt_metrics = list()
            if not get_balanced_acc and not get_auc:
                metrics = None
            else:                
                if get_balanced_acc:
                    opt_metrics.append('balanced_accuracy')
                if get_auc:
                    opt_metrics.append('auc')                    
                metrics = Metrics(opt_metrics)                


            for data in data_loader:

                # In data we may have imgs, labels and extra info. If extra info is [], it means we don't have it
                # for the this training case. Imgs came in data[0], labels in data[1] and extra info in data[2]
                try:
                    images_batch, labels_batch, meta_data_batch, _ = data
                    labels_batch = torch.tensor(labels_batch)
                except ValueError:
                    images_batch, labels_batch = data
                    meta_data_batch = []

                if len(meta_data_batch):
                    # Moving the data to the deviced that we set above
                    images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                    meta_data_batch = meta_data_batch.to(device)
                    meta_data_batch = meta_data_batch.float()

                    # Doing the forward pass using meta_data
                    pred_batch = model(images_batch, meta_data_batch)
                else:
                    # Moving the data to the device that we set above
                    images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

                    # Doing the forward pass without using meta-data
                    pred_batch = model(images_batch)

                # Computing the loss
                L = loss_fn(pred_batch, labels_batch)

                batch_sz = labels_batch.size(0)

                # Computing the accuracy
                _, pred = pred_batch.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(labels_batch.view(1, -1).expand_as(pred))
                # print(correct.shape)
                total_correct = correct.view(-1).float().sum(0)
                acc = total_correct.mul_(100/batch_sz)

                
                
                # acc, topk_acc = accuracy(pred_batch, labels_batch, topk=(1, topk))

                loss_avg.update(L.item())
                acc_avg.update(acc.item())
                # topk_avg.update(topk_acc.item())

                if metrics is not None:
                    # Moving the data to CPU and converting it to numpy in order to compute the metrics
                    pred_batch_np = nnF.softmax(pred_batch, dim=1).cpu().numpy()
                    labels_batch_np = labels_batch.cpu().numpy()
                    # updating the scores
                    metrics.update_scores(labels_batch_np, pred_batch_np)

                # Updating tqdm
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

    bal_acc, auc = None, None
    if metrics is not None:
        # Getting the metrics
        metrics.compute_metrics()
        try:
            bal_acc =  metrics.metrics_values['balanced_accuracy']
        except KeyError:
            bal_acc = None        
        try:
            auc =  metrics.metrics_values['auc']
        except KeyError:
            auc = None

    return {"loss": loss_avg(), "accuracy": acc_avg(), "balanced_accuracy": bal_acc, 'auc': auc}


# Testing the model
def test_model (model, data_loader, checkpoint_path=None, loss_fn=None, device=None, save_pred=False,
                    partition_name='Test', metrics_to_comp=('accuracy'), class_names=None, metrics_options=None,
                    apply_softmax=True, verbose=True, full_path_pred=None):
    """
    This function evaluates a given model for a given data_loader

    :param model (nn.Model): the model you'd like to evaluate
    :param data_loader (DataLoader): the DataLoader containing the data partition
    :param checkpoint_path(string, optional): string with a checkpoint to load the model. If None, no checkpoint is
    loaded. Default is None.
    :param loss_fn (nn.Loss): the loss function used in the training
    :param partition_name (string): the partition name
    :param metrics (list, tuple or string): it's the variable that receives the metrics you'd like to compute.
        IMPORTANT: if metrics is None, only the prediction.csv will be generated. Default is only the accuracy.
    :param class_names (list, tuple): a list or tuple containing the classes names in the same order you use in the
        label. For ex: ['C1', 'C2']. For more information about the options, please, refers to
        jedy.pytorch.model.metrics.py
    :param metrics_ options (dict): this is a dict containing some options to compute the metrics. Default is None.
    For more information about the options, please, refers to jedy.pytorch.model.metrics.py
    :param device (torch.device, optional): the device to use. If None, the code will look for a device. Default is
    None. For more information about the options, please, refers to jedy.pytorch.model.metrics.py
    :param verbose (bool, optional): if you'd like to print information o the screen. Default is True
    True. Default is False.

    :return: a instance of the classe metrics
    """

    # setting the model to evaluation mode
    model.eval()

    def _get_predictions (model, images_batch, meta_data_batch=None):        
        with torch.no_grad():
            if meta_data_batch is None:
                pred_batch = model(images_batch)
            else:
                pred_batch = model(images_batch, meta_data_batch)
        return pred_batch

    if checkpoint_path is not None:
        model = load_model(checkpoint_path, model)

    if device is None:
        # Setting the device
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            device = torch.device("cpu")

    # Moving the model to the given device
    model.to(device)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # Setting the metrics object
    metrics = Metrics (metrics_to_comp, class_names, metrics_options)

    print("Testing...")
    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, ncols=100) as t:

        loss_avg = AVGMetrics()

        for data in data_loader:
            # In data we may have images, labels, meta-data, and image ID. If meta-data is [], it means we don't have it
            # for the this training case. Images came in data[0], labels in data[1], meta_data in data[2], and image_id
            # in data[3]
            try:
                images_batch, labels_batch, meta_data_batch, img_id = data
            except ValueError:
                images_batch, labels_batch = data
                meta_data_batch = []
                img_id = None

            if len(meta_data_batch):
                # Moving the data to the device that we set above
                images_batch = images_batch.to(device)
                if len(labels_batch):
                    labels_batch = labels_batch.to(device)
                meta_data_batch = meta_data_batch.to(device)
                meta_data_batch = meta_data_batch.float()

                # Doing the forward pass using meta-data
                pred_batch = _get_predictions (model, images_batch, meta_data_batch)
            elif len(labels_batch):
                # Moving the data to the device that we set above
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

                # Doing the forward pass without using meta-data
                pred_batch = _get_predictions(model, images_batch)
            else:
                # Moving the data to the deviced that we set above
                images_batch = images_batch.to(device)

                # Doing the forward pass without using meta_data
                pred_batch = _get_predictions(model, images_batch)

            # Computing the loss
            if len(labels_batch):
                L = loss_fn(pred_batch, labels_batch)
                loss_avg.update(L.item())
                labels_batch_np = labels_batch.cpu().numpy()
            else:
                labels_batch_np = None
                loss_avg.update(0)

            # Moving the data to CPU and converting it to numpy in order to compute the metrics
            if apply_softmax:
                pred_batch_np = nnF.softmax(pred_batch,dim=1).cpu().numpy()
            else:
                pred_batch_np = pred_batch.cpu().numpy()

            # updating the scores
            metrics.update_scores(labels_batch_np, pred_batch_np, img_id)

            # Updating tqdm
            if metrics.metrics_names is None:
                t.set_postfix(loss='{:05.3f}'.format(0.0))
            else:
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

            t.update()

    # Adding loss into the metric values
    metrics.add_metric_value("loss", loss_avg())

    # Getting the metrics
    metrics.compute_metrics()

    if save_pred or metrics.metrics_names is None:
        if full_path_pred is None:
            metrics.save_scores()
        else:
            _spt = full_path_pred.split('/')
            _folder = "/".join(_spt[0:-1])
            _p = _spt[-1]
            metrics.save_scores(folder_path=_folder, pred_name=_p)

    if verbose:
        print('- {} metrics:'.format(partition_name))
        metrics.print()


    return metrics.metrics_values


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the CNN start phase

If you find any bug or have some suggestion, please, email me.
"""

# tensorboardX may be replaced by torch.utils.tensorboard. However, it's throwing some warnings about compatibility
# between tensorflow and numpy. While they don't find a solution, I'll keep using tensorboardX to avoid this warning
# on the screen. Also, both libs works fine. Note: the warning will raise when you run the tensorboard command in the
# folder

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from raug.raug.checkpoints import load_model, save_model
# from raug.eval import metrics_for_eval
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from raug.raug.metrics import accuracy, TrainHistory
from raug.raug.utils.classification_metrics import AVGMetrics
from raug.raug.utils.telegram_bot import TelegramBot
import logging
import time


def _config_logger(save_path, file_name):
    """
        Internal function to configure the logger
    """
    logger = logging.getLogger("Train-Logger")
    # Checking if the folder logs doesn't exist. If True, we must create it.
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    logger_filename = os.path.join(save_path, f"{file_name}_{str(time.time()).replace('.','')}.log")
    fhandler = logging.FileHandler(filename=logger_filename, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def _train_epoch (model, optimizer, loss_fn, data_loader, c_epoch, t_epoch, device, topk=2):
    """
    This function trains an epoch of the dataset, that is, it goes through all dataset batches once.
    :param model (torch.nn.Module): a module to be trained
    :param optimizer (torch.optim.optim): an optimizer to fit the model
    :param loss_fn (torch.nn.Loss): a loss function to evaluate the model prediction
    :param data_loader (torch.utils.DataLoader): a dataloader containing the dataset
    :param c_epoch (int): the current epoch
    :param t_epoch (int): the total number of epochs
    :param device (torch.device): the device to carry out the training 
    """

    # setting the model to training mode
    model.train()

    print ("Training...")
    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, desc='Epoch {}/{}: '.format(c_epoch, t_epoch), ncols=100) as t:


        # Variables to store the avg metrics
        loss_avg = AVGMetrics()
        acc_avg = AVGMetrics()
        topk_avg = AVGMetrics()

        # Getting the data from the DataLoader generator
        for batch, data in enumerate(data_loader, 0):

            # print (data)
            # In data we may have imgs, labels and extra info. If extra info is [], it means we don't have it
            # for the this training case. Imgs came in data[0], labels in data[1] and extra info in data[2]
            try:
                imgs_batch, labels_batch, metadata_batch, _ = data
            except ValueError:
                imgs_batch, labels_batch = data
                metadata_batch = []


            if len(metadata_batch):
                # In this case we have extra information and we need to pass this data to the model
                # Moving the data to the deviced that we set above
                imgs_batch, labels_batch = imgs_batch.to(device), labels_batch.to(device)
                metadata_batch = metadata_batch.to(device)
                metadata_batch = metadata_batch.float()

                # Doing the forward pass
                out = model(imgs_batch, metadata_batch)
            else:
                # In this case we don't have extra info, so the model doesn't expect for it
                # Moving the data to the deviced that we set above
                imgs_batch, labels_batch = imgs_batch.to(device), labels_batch.to(device)
                # print (imgs_batch.size())

                # Doing the forward pass
                out = model(imgs_batch)
                # print (out)

            batch_sz = labels_batch.size(0)
            # Computing loss function
            loss = loss_fn(out, labels_batch)

            _, pred = out.topk(1, 1, True, True)
            # print(labels_batch)
            # print(pred)
            # print(labels_batch.view(-1, 1).expand_as(pred))
            pred = pred.t()
            correct = pred.eq(labels_batch.view(1, -1).expand_as(pred))
            # print(correct.shape)
            total_correct = correct.view(-1).float().sum(0)
            acc = total_correct.mul_(100/batch_sz)

            # break

            # Computing the accuracy
            # acc, topk_acc = accuracy(out, labels_batch, topk=(1, topk))

            # Getting the avg metrics
            loss_avg.update(loss.item())
            acc_avg.update(acc.item())
            # topk_avg.update(topk_acc.item())

            # Zero the parameters gradient
            optimizer.zero_grad()

            # Computing gradients and performing the update step
            loss.backward()
            optimizer.step()

            # Updating tqdm
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # return {"loss": loss_avg(), "accuracy": acc_avg(), "topk_acc": topk_avg() }
    return {"loss": loss_avg(), "accuracy": acc_avg()}

def fit_model (model, train_data_loader, val_data_loader, optimizer=None, loss_fn=None, epochs=10,
               epochs_early_stop=None, save_folder=None, initial_model=None, best_metric="loss", device=None,
               topk=2, schedule_lr=None, config_bot=None, model_name="CNN", resume_train=False, history_plot=True,
               val_metrics=('balanced_accuracy', 'auc'), metric_early_stop=None):
    """
    This is the main function to carry out the training phase.

    :param c_epoch (int): the current epoch
    :param t_epoch (int): the total number of epochs
    :param device (torch.device): the device to carry out the training
    :param model (torch.nn.Module): a module to be trained
    :param train_data_loader (torch.utils.DataLoader): a dataloader containing the start dataset
    :param val_data_loader (torch.utils.DataLoader): a dataloader containing the validation dataset
    :param optimizer (torch.optim.optim, optional): an optimizer to fit the model. If None, it will use the
     optim.Adam(model.parameters(), lr=0.001). Default is None.
    :param loss_fn (torch.nn.Loss, optional): a loss function to evaluate the model prediction. If None, it will use the
    nn.CrossEntropyLoss(). Default is None.
    :param epochs (int, optional): the number of epochs to start the model. Default is 10.
    :param epochs_early_stop (int, optional): if you'd like to check early stop, pass the number of epochs that need to
    be achieved to stop the training. It checks if the loss is improving. If it doesn't improve for epochs_early_stop,
    training stops. If None, the training is never stopped. Default is None.
    :param save_folder (string, optional): if you'd like to save the last and best checkpoints, just pass the folder
    path in which the checkpoint will be saved. If None, the model is not saved in the disk. Default is None.
    :param initial_model (string, optinal): if you'd like to restart the training from a given saved checkpoint, pass
    the path to this file here. If None, the model starts training from scratch. Default is None.
    :param resume_train (bool, optional): if you'd like to resume the training using the last values for optimizer and
    starting from the last epoch trained, set it True. Default is False.
    :param class_names (list, optional): the list of class names.
    :param best_metric (string, optional): if you chose save the model, you can inform the metric you'd like to save as
    the best. Default is loss.
    :param device (torch.device): the device you'd like to start the model. If None, it will check if you have a GPU
    available. If not, it use the CPU. Default is None.
    :param topk: number of top accuracies to compute
    :param schedule_lr (bool, optional): If you're using a schedule for the learning rate you need to pass it using this
    variable. If this is None, no schdule will be performed. Default is None.
    :param config_bot (string or dictionary, optional): this is a string containing the chat_id for the bot or a dict
    containing the chat_id and the token, example: {chat_id: xxx, token: yyy}. If None, the chat_bot will not be used.
    Default is None.
    :param model_name (string, optional): this is the model's name, ex: ResNet. Defaul is CNN.
    """

    logger = _config_logger(save_folder, model_name)
    logger.info("Starting the training phase")

    if epochs_early_stop is not None:
        logger.info('Early stopping is set using the number of epochs without improvement')
    if metric_early_stop is not None:
        logger.info('Early stopping is set using the min/max metric as threshold')
    if epochs_early_stop is None and metric_early_stop is None:
        logger.info('No early stopping is set')

    history = TrainHistory()

    # Configuring the Telegram bot
    tele_bot = None
    if config_bot is not None:
        logger.info('Using TelegramBot to track the training')
        if isinstance(config_bot, str):
            tele_bot = TelegramBot(config_bot, model_name=model_name)
        elif isinstance(config_bot, dict):
            config_bot["model_name"] = model_name
            tele_bot = TelegramBot(**config_bot)
        else:
            logger.error("There is a problem in config_bot variable")
            raise Exception("- The config_bot is not ok. Check it, please!")

    if loss_fn is None:
        logger.info('Loss was set as None. Using the CrossEntropy as default')
        loss_fn = nn.CrossEntropyLoss()

    if optimizer is None:
        logger.info('Optimizer was set as None. Using the Adam with lr=0.001 as default')
        optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Checking if we have a saved model. If we have, load it, otherwise, let's start the model from scratch
    epoch_resume = 0
    if initial_model is not None:
        logger.info("Loading the saved model in {} folder".format(initial_model))

        if resume_train:
            model, optimizer, loss_fn, epoch_resume = load_model(initial_model, model,opt_fn=optimizer,loss_fn=loss_fn)
            logger.info("Resuming the training from epoch {} ...".format(epoch_resume))
        else:
            model = load_model(initial_model, model)

    else:
        logger.info("The model will be trained from scratch")


    # Setting the device(s)
    # If GPU is available, let's move the model to there. If you have more than one, let's use them!
    m_gpu = 0
    if device is None:
        if torch.cuda.is_available():

            device = torch.device("cuda")
            # device = torch.device("cuda:" + str(torch.cuda.current_device()))
            m_gpu = torch.cuda.device_count()
            if m_gpu > 1:
                logger.info("The training will be carry out using {} GPUs:".format(m_gpu))
                for g in range(m_gpu):
                    logger.info(torch.cuda.get_device_name(g))

                model = nn.DataParallel(model)
            else:
                logger.info("The training will be carry out using 1 GPU:")
                logger.info(torch.cuda.get_device_name(0))
        else:
            logger.info("The training will be carry out using CPU")
            device = torch.device("cpu")
    else:
        logger.info("The training will be carry out using 1 GPU:")
        logger.info(torch.cuda.get_device_name(device))


    # Moving the model to the given device
    model.to(device)

    # Setting data to store the best mestric
    logging.info("The best metric to get the best model will be {}".format(best_metric))
    if best_metric == 'loss':
        best_metric_value = 1000
    else:
        best_metric_value = 0
    best_flag = False

    # Checking if we need to compute the balanced accuracy
    if val_metrics is None:
        get_bal_acc = False
        get_auc = False
    else:
        get_bal_acc = 'balanced_accuracy' in val_metrics
        get_auc = 'auc' in val_metrics

    # setting a flag for the early stop
    early_stop_count = 0
    best_epoch = 0

    # writer is used to generate the summary files to be loaded at tensorboard
    writer = SummaryWriter (os.path.join(save_folder, 'summaries'))

    if tele_bot is not None:
        tele_bot.start_bot()

        # Checking the GPU names or if it's gonna run in a CPU
        msg = ""
        if m_gpu == 0:
            msg = "--------\nThe training will be executed in CPU\n--------"
        else:
            msg = "--------\nThe the training will be executed in the following GPU(s):\n"
            for g in range(m_gpu):
                msg += torch.cuda.get_device_name(g) + "\n"
            msg += "--------"

        tele_bot.send_msg(msg)

    # Let's iterate for `epoch` epochs or a tolerance.
    # It always start from epoch resume. If it's set, it starts from the last epoch the training phase was stopped,
    # otherwise, it starts from 0
    epoch = epoch_resume
    while epoch < epochs:

        # Updating epoch
        epoch += 1

        # Training and getting the metrics for one epoch
        train_metrics = _train_epoch(model, optimizer, loss_fn, train_data_loader, epoch, epochs, device, topk)

        # After each epoch, we evaluate the model for the training and validation data
        val_metrics = metrics_for_eval (model, val_data_loader, device, loss_fn, topk,
                                        get_balanced_acc=get_bal_acc, get_auc=get_auc)

        # Checking the schedule if applicable
        if isinstance(schedule_lr, torch.optim.lr_scheduler.ReduceLROnPlateau):
            schedule_lr.step(best_metric_value)
        elif isinstance(schedule_lr, torch.optim.lr_scheduler.MultiStepLR):
            schedule_lr.step(epoch)

        # Getting the current LR
        current_LR = None
        for param_group in optimizer.param_groups:
            current_LR = param_group['lr']


        writer.add_scalars('Loss', {'val-loss': val_metrics['loss'],
                                                 'start-loss': train_metrics['loss']},
                                                 epoch)

        writer.add_scalars('Accuracy', {'val-loss': val_metrics['accuracy'],
                                    'start-loss': train_metrics['accuracy']},
                                    epoch)

        history.update(train_metrics['loss'], val_metrics['loss'], train_metrics['accuracy'], val_metrics['accuracy'])


        # Getting the metrics for the training partition epoch
        train_print = "-- Loss: {:.3f}\n-- Acc: {:.3f}\n".format(train_metrics["loss"],
                                                                               train_metrics["accuracy"])
                                                                              #  topk, train_metrics["topk_acc"])

        # Getting the metrics for the validation partition in this epoch
        val_print = "-- Loss: {:.3f}\n-- Acc: {:.3f}\n".format(val_metrics["loss"],
                                                                              val_metrics["accuracy"])
                                                                              # topk, val_metrics["topk_acc"])
        if get_bal_acc:
            val_print += "\n-- Balanced accuracy: {:.3f}".format(val_metrics['balanced_accuracy'])
        if get_auc:
            val_print += "\n-- AUC: {:.3f}".format(val_metrics['auc'])


        early_stop_count += 1
        new_best_print = None
        # Defining the best metric for validation
        if best_metric == 'loss':
            if val_metrics[best_metric] <= best_metric_value:
                best_metric_value = val_metrics[best_metric]
                new_best_print = '\n-- New best {}: {:.3f}'.format(best_metric, best_metric_value)
                best_flag = True
                best_epoch = epoch
                early_stop_count = 0
        else:
            if val_metrics[best_metric] >= best_metric_value:
                best_metric_value = val_metrics[best_metric]
                new_best_print = '\-- New best {}: {:.3f}'.format(best_metric, best_metric_value)
                best_flag = True
                best_epoch = epoch
                early_stop_count = 0

        # Check if it's the best model in order to save it
        if save_folder is not None:
            save_model(model, save_folder, epoch, optimizer, loss_fn, best_flag,val_metrics[best_metric] ,multi_gpu=m_gpu > 1)
        best_flag = False

        # Updating the logger
        msg = "Metrics for epoch {} out of {}\n".format(epoch, epochs)
        msg += "- Train\n"
        msg += train_print + "\n"
        msg += "\n- Validation\n"
        msg += val_print + "\n"
        msg += "\n- Training info"
        msg += "\n-- Early stopping counting: {} max to stop is {}".format(early_stop_count, epochs_early_stop)
        msg += "\n-- Current LR: {}".format(current_LR)
        if new_best_print is not None:
            msg += new_best_print
        msg += "\n-- Best {} so far: {:.3f} on epoch {}\n".format(best_metric, best_metric_value, best_epoch)

        # Updating the bot
        if tele_bot is not None:
            msg_best = "The best {} for the validation set so far is {:.3f} on epoch {}".format(best_metric,
                                                                                                best_metric_value,
                                                                                                best_epoch)
            tele_bot.best_info = msg_best

            if tele_bot.info:
                tele_bot.send_msg(msg)

            tele_bot.current_epoch = "The current training epoch is {} out of {} and the current LR is {}".format(epoch, epochs, current_LR)

        # Checking the early stop
        if epochs_early_stop is not None:
            if early_stop_count >= epochs_early_stop:
                logger.info(msg)
                logger.info("The early stop trigger was activated. The validation {} " .format(best_metric) +
                            "{:.3f} did not improved for {} epochs.".format(best_metric_value,
                                                                            epochs_early_stop) +
                            "The training phase was stopped.")

                break

        # Checking the early stop
        if metric_early_stop is not None:
            stop = False
            if best_metric == 'loss':
                if metric_early_stop >= best_metric_value:
                    stop = True
            else:
                if metric_early_stop <= best_metric_value:
                    stop = True

            if stop:
                logger.info(msg)
                logger.info("The early stop trigger was activated. The validation {} ".format(best_metric) +
                            "{:.3f} achieved the defined threshold {:.3f}.".format(best_metric_value,
                                                                            metric_early_stop) +
                            "The training phase was stopped.")
                break

        # Sending all message to the logger
        logger.info(msg)

    # Closing the bot
    if tele_bot is not None:
        msg_bot = "--------\nThe trained is finished!\n"
        msg_bot += "The best {} founded for the validation set was {:.3f} on epoch {}\n".format(best_metric,
                                                                                            best_metric_value,
                                                                                            best_epoch)
        msg_bot += "See you next time :)\n--------\n"
        tele_bot.send_msg(msg_bot)
        tele_bot.stop_bot()

    if history_plot:
        history.save_plot(save_folder)

    history.save(save_folder)
    print('\n')

    writer.close()