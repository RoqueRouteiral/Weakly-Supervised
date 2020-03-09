import os
import time
import torch
import torch.nn as nn
import numpy as np
import logging

# Segmentation models
from tools.plots import plot_history
import matplotlib.image as mima
import matplotlib.pyplot as plt
from models.resnet import resnet34
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Build the model
class Model():
    
    def __init__(self, cf):
        self.cf = cf
        self.savepath = self.cf['experiments']+self.cf['exp_name']
        self.device = torch.device("cuda:"+str(self.cf['gpu']) if torch.cuda.is_available() else "cpu")
        self.model = resnet34()
        self.model.to(self.device)
        # Output the model
        logging.info('   Model: ' + self.cf['model_name'])
         

    
    def train(self, train_gen, val_gen):

        self.criterion = nn.BCELoss()
        if self.cf['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.cf['learning_rate'] )
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.cf['learning_rate'] )
        
        #Starting the training
        train_losses=np.zeros(self.cf['epochs'])
        train_accs=np.zeros(self.cf['epochs'])
        val_losses=np.zeros(self.cf['epochs'])
        val_accs=np.zeros(self.cf['epochs'])
        best_loss = np.inf
        for epoch in range(self.cf['epochs']):
            start_time = time.time()
            train_losses[epoch], train_accs[epoch]=self._one_epoch(train_gen,'train')
            val_losses[epoch], val_accs[epoch]=self._one_epoch(val_gen,'eval')                    
            logging.info('[%d/%d] loss: %.3f Acc %.3f val_loss: %.3f val_Acc: %.3f time: %.3f min' % (epoch + 1, self.cf['epochs'], train_losses[epoch], train_accs[epoch], val_losses[epoch], val_accs[epoch], (time.time() - start_time)/60))
            if not epoch%self.cf['snapshots']:
                torch.save(self.model.state_dict(), (self.savepath+'/snapshot_epoch{}.ckpt'.format(epoch)))
                if epoch>0:
                    plot_history(train_losses[0:epoch], train_accs[0:epoch], val_losses[0:epoch], val_accs[0:epoch], self.savepath+'/hist_epoch{}.png'.format(epoch))
            if val_losses[epoch]<best_loss:
                best_loss = val_losses[epoch]
                torch.save(self.model.state_dict(), (self.savepath+'/snapshot_epoch{}.ckpt'.format(epoch)))
        torch.save(self.model.state_dict(), self.savepath+'/trained_model.ckpt')
        #saving epochs
        np.save(self.savepath+'/train_losses.npy',train_losses)
        np.save(self.savepath+'/train_accs.npy',train_accs)
        np.save(self.savepath+'/val_losses.npy',val_losses)
        np.save(self.savepath+'/val_accs.npy',val_accs)
        
        plot_history(train_losses, train_accs, val_losses, val_accs, self.savepath+'/hist.png')
        print('Finished training')
        
    def test(self, test_gen):
        self.model.load_state_dict(torch.load(self.cf['weights_test_file']))
        running_acc = []
        self.model.eval()
        with torch.no_grad(): 
            for i, (data, patient_name) in enumerate(test_gen, 0):
                    # get the inputs
                    inputs, labels = data
                    # forward
                    outputs = self.model(inputs.to(self.device))>0.5
                    for image in range(inputs.shape[0]):
                        running_acc.append(accuracy_score(labels.cpu(), (outputs>0.5).cpu()))
                        logging.info('-Patient {}-: Acc={}'.format(patient_name[image],running_acc[-1]))
            logging.info('Acc: %.3f'%(np.array(running_acc).mean()))
            np.save(self.savepath+'/accs.npy',running_acc)
        print('Finished testing')
   
                            
    def _one_epoch(self, loader, phase):
        with torch.set_grad_enabled(phase=='train'):
            if phase=='train':
                self.model.train()
            else:
                self.model.eval()
            running_loss = 0.0
            running_acc = 0.0
            for i, (data, _) in enumerate(loader, 0):
                # get the inputs
                inputs, labels = data
                # forward + backward + optimize
                outputs = (self.model(inputs.to(self.device)))
                loss = self.criterion(outputs.cpu().float(), labels.cpu().float())
                if phase=='train':
                    self.optimizer.zero_grad()               
                    loss.backward()
                    self.optimizer.step()
                acc = accuracy_score(labels.cpu().float(), (outputs>0.5).cpu().float())
                # print statistics
                running_loss += loss.item() * inputs.shape[0] / self.cf['batch_size'] #weight for incomplete last batch
                running_acc += acc.item() * inputs.shape[0] / self.cf['batch_size']
            epoch_loss=running_loss/len(loader)
            epoch_acc=running_acc/len(loader)
        return epoch_loss, epoch_acc