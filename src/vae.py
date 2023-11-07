# vae.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from load import load_data  # Make sure this function is correctly implemented
from preprocess import preprocess_data  # Make sure this function is correctly implemented

# Define the Encoder network
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        z_mean = self.mean(x)
        z_logvar = self.logvar(x)
        return z_mean, z_logvar

# Define the Decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, z):
        z = torch.relu(self.linear1(z))
        z = torch.relu(self.linear2(z))
        reconstructed = torch.sigmoid(self.out(z))
        return reconstructed

# Define the VAE class
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(z_mean)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_logvar

    def loss_function(self, recon_x, x, z_mean, z_logvar):
        # Reconstruction loss
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        # Total loss
        return BCE + KLD


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
raw_data = load_data('/Users/jasmingadestovlbaek/Documents/DeepLearning/Project/Project/_raw/archs4_gene_expression_norm_transposed.tsv.gz')  # Adjust the path accordingly
preprocessed_data = preprocess_data(raw_data)

# Assuming preprocessed_data is a NumPy array
x_train = torch.tensor(preprocessed_data, dtype=torch.float32)

# If your dataset includes labels or you want to include another dimension, adjust as necessary
# Here, we're assuming an unsupervised scenario as typical with autoencoders
train_dataset = TensorDataset(x_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Parameters
input_dim = x_train.shape[1]
latent_dim = 64  # The latent space dimensionality you wish to use
output_dim = input_dim  # Output dimension should match input dimension for reconstruction

# Create the VAE
encoder = Encoder(input_dim, latent_dim)
decoder = Decoder(latent_dim, output_dim)
model = VAE(encoder, decoder)
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for x, in train_loader:
        x = x.to(device)
        recon_x, z_mean, z_logvar = model(x)
        loss = model.loss_function(recon_x, x, z_mean, z_logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Compute average loss for the epoch
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch}, Loss: {train_loss}")

# Save the model
torch.save(model.state_dict(), '/Users/jasmingadestovlbaek/Documents/DeepLearning/Project/Project/results')  # Adjust the path accordingly
