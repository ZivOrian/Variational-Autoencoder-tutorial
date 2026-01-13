import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self,device):
        super(VAE,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = 48#64
        self.common_fc = nn.Sequential(
            #nn.Linear(28*28, out_features=392), nn.Tanh(),
            nn.Linear(28*28, out_features=512), nn.LeakyReLU(0.2),
            nn.Linear(512, out_features=256), nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, out_features=128),#nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128)
        )
        
        self.mean_fc = nn.Sequential(
            nn.Linear(128, out_features=self.latent_dim)
            #nn.Linear(48, out_features=self.latent_dim)
        )
        
        self.log_var_fc = nn.Sequential(
            #nn.Linear(128, out_features=48), nn.Tanh(),
            nn.Linear(128, out_features=self.latent_dim)
        )
        
        self.decoder_fcs = nn.Sequential(
            #nn.Linear(self.latent_dim, out_features=48), nn.Tanh(), 
            nn.Linear(self.latent_dim, out_features=128), nn.LeakyReLU(0.2),
            nn.Linear(128, out_features=256),nn.LeakyReLU(0.2),
            nn.Linear(256,out_features=512),nn.LeakyReLU(0.2),
            nn.Linear(512, out_features=28*28),nn.Sigmoid()
        )
        
        self.to(self.device)

    def encode(self,x):
        # B,C,H,W
        flat_x = torch.flatten(x, start_dim=1)
        out = self.common_fc(flat_x)
        mean = self.mean_fc(out)
        log_var = self.log_var_fc(out)

        log_var = torch.clamp(log_var, min=-10, max=10)

        return mean, log_var


    def sample(self, mean, log_var):
        # Calculate standard deviation
        std = torch.exp(0.5 * log_var)
        
        # Generate noise (epsilon)
        # We do NOT need gradients for the noise itself
        eps = torch.randn_like(std)
        
        # Reparameterization trick
        z = eps * std + mean
        return z
    
    
    def decode(self, z):
        out = self.decoder_fcs(z).to(self.device)
        #out = torch.reshape(out, [1, 28*28])
        return out
        
    
    def forward(self, batch_x):
        #B,C,H,W
        outputs = []
        logv_arr = []
        mean_arr = []
        #Encoder
        #batch_x = self.bn2d(batch_x)
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


    def generate(self, n_samples=1):
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        with torch.no_grad():
            samples = self.decode(z)
        return samples.view(-1, 1, 28, 28)
    

    def test_generation_quality(model, n_samples=16):
        """
        Generate samples and check if they're diverse
        """
        samples = model.generate(n_samples)
        
        # Check variance across samples
        variance = torch.var(samples)
        print(f"\nGenerated {n_samples} samples")
        print(f"Variance across samples: {variance:.4f}")
        print("Expected range: 10-50 for good diversity")
        
        # Check if outputs are not all the same
        pairwise_diff = []
        for i in range(min(5, n_samples)):
            for j in range(i+1, min(5, n_samples)):
                diff = torch.mean((samples[i] - samples[j])**2)
                pairwise_diff.append(diff.item())
        
        avg_diff = sum(pairwise_diff) / len(pairwise_diff)
        print(f"Average pairwise MSE: {avg_diff:.4f}")
        print("Expected: > 0.01 for diverse outputs")
        
        return samples