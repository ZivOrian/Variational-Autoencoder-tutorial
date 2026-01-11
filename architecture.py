import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        
        
        self.common_fc = nn.Sequential(
            nn.Linear(28*28, out_features=196), nn.Sigmoid(),
            nn.Linear(196, out_features=48), nn.Sigmoid(),
        )
        
        self.mean_fc = nn.Sequential(
            nn.Linear(48, out_features=16), nn.Sigmoid(),
            nn.Linear(16, out_features=2), nn.Sigmoid()
        )
        
        self.log_var_fc = nn.Sequential(
            nn.Linear(48, out_features=16), nn.Sigmoid(),
            nn.Linear(16, out_features=2), nn.Sigmoid()
        )
        
        self.decoder_fcs = nn.Sequential(
            nn.Linear(2, out_features=16), nn.Sigmoid(), 
            nn.Linear(16, out_features=48), nn.Sigmoid(),
            nn.Linear(48, out_features=196), nn.Sigmoid(),
            nn.Linear(196, out_features=28*28), nn.Sigmoid(),
        )
        

    def encode(self,x):
        # B,C,H,W
        flat_x = torch.flatten(x, start_dim=1)
        out = self.common_fc(flat_x)
        mean = self.mean_fc(out)
        log_var = self.log_var_fc(out)
        return mean, log_var


    def sample(self, mean, log_var):
        std = torch.exp(0.5*torch.flatten(log_var, start_dim=-1))
        z = torch.randn_like(torch.flatten(std, start_dim=-1)
                             ,requires_grad=True)
        return z * std + mean
    
    
    def decode(self, z):
        out = self.decoder_fcs(z)
        #out = torch.reshape(out, [1, 28*28])
        return out
        
    
    def forward(self, batch_x):
        #B,C,H,W
        outputs = []
        logv_arr = []
        mean_arr = []
        #Encoder
        mean, log_var = self.encode(batch_x)
        #Sampling
        z = self.sample(mean,log_var)
        #Decoder
        logv_arr.append(log_var)
        mean_arr.append(mean)
        outputs.append(self.decode(z))

        mean_arr = torch.stack(mean_arr, dim=0)
        logv_arr = torch.stack(logv_arr, dim=0)
        out = torch.stack(outputs, dim=0)
        return mean_arr, logv_arr, out


    def generate(self,device):
        n_sample = torch.normal(
            mean=0.,std=1.,size=(2,),requires_grad=False).to(device) # A sample from the standard normal distribution
        
        gen = self.decode(n_sample)
        return gen