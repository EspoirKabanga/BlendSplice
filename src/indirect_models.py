import torch.nn as nn
import torch.optim as optim

class SpliceRover(nn.Module):
    def __init__(self):
        super(SpliceRover, self).__init__()
        self.name = 'SpliceRover'
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=70, kernel_size=9, 
                     padding='same'),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=70, out_channels=100, kernel_size=7,
                     padding='same'),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=7,
                     padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),  # Default stride equals kernel_size
            nn.Dropout(0.2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=200, kernel_size=7,
                     padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=200, out_channels=250, kernel_size=7,
                     padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2)
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(250 * 8, 512),  # Original length maintained due to stride=1
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
            # nn.Softmax(dim=1)
            )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        
        return x


class SpliceFinder(nn.Module):
    def __init__(self):
        super(SpliceFinder, self).__init__()
        self.name = 'SpliceFinder'
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=50, kernel_size=9, 
                     padding='same'),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        self.dense = nn.Sequential(
            nn.Linear(50 * 402, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 2),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class DeepSplicer(nn.Module):
    def __init__(self):
        super(DeepSplicer, self).__init__()
        self.name = 'DeepSplicer'
        
        # Three identical convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=50, kernel_size=9, 
                     padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=9,
                     padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=9,
                     padding='same'),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        self.dense = nn.Sequential(
            nn.Linear(50 * 402, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 2),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
class IntSplice(nn.Module):
   def __init__(self):
       super(IntSplice, self).__init__()
       self.name = 'IntSplice'

       # First conv block
       self.conv1 = nn.Sequential(
           nn.Conv1d(in_channels=4, out_channels=64, kernel_size=10, 
                    padding='same'),
           nn.ReLU(),
           nn.MaxPool1d(kernel_size=2),
           nn.Dropout(0.3)
       )

       # Second conv block
       self.conv2 = nn.Sequential(
           nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,
                    padding='same'),
           nn.ReLU(), 
           nn.MaxPool1d(kernel_size=2),
           nn.Dropout(0.2)
       )

       # Third conv block
       self.conv3 = nn.Sequential(
           nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,
                    padding='same'),
           nn.ReLU(),
           nn.MaxPool1d(kernel_size=2),
           nn.Dropout(0.3)
       )

       # Fourth conv block
       self.conv4 = nn.Sequential(
           nn.Conv1d(in_channels=256, out_channels=512, kernel_size=2,
                    padding='same'),
           nn.ReLU(),
           nn.MaxPool1d(kernel_size=2),
           nn.Dropout(0.2)
       )

       self.flatten = nn.Flatten()

       # Dense layers
       self.dense = nn.Sequential(
           nn.Linear(512 * 25, 512),  # 402/2^4 from maxpooling ≈ 25
           nn.ReLU(),
           nn.Dropout(0.2),
           nn.Linear(512, 2),
        #    nn.Softmax(dim=1)
       )
   
   def forward(self, x):
       x = x.transpose(1, 2)
       x = self.conv1(x)
       x = self.conv2(x)
       x = self.conv3(x)
       x = self.conv4(x)
       x = self.flatten(x)
       x = self.dense(x)
       return x
   
class Spliceator(nn.Module):
   def __init__(self):
       super(Spliceator, self).__init__()
       self.name = 'Spliceator'

       # First conv block
       self.conv1 = nn.Sequential(
           nn.Conv1d(in_channels=4, out_channels=16, kernel_size=7,
                    padding='same'),
           nn.ReLU(),
           nn.MaxPool1d(kernel_size=2),
           nn.Dropout(0.2)
       )

       # Second conv block
       self.conv2 = nn.Sequential(
           nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6,
                    padding='same'),
           nn.ReLU(),
           nn.MaxPool1d(kernel_size=2),
           nn.Dropout(0.2)
       )

       # Third conv block
       self.conv3 = nn.Sequential(
           nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6,
                    padding='same'),
           nn.ReLU(),
           nn.MaxPool1d(kernel_size=2),
           nn.Dropout(0.2)
       )

       self.flatten = nn.Flatten()

       # Dense layers
       self.dense = nn.Sequential(
           nn.Linear(64 * 50, 100),  # 402/2^3 from maxpooling ≈ 50
           nn.ReLU(),
           nn.Linear(100, 2),
        #    nn.Softmax(dim=1)
       )
   
   def forward(self, x):
       x = x.transpose(1, 2)
       x = self.conv1(x)
       x = self.conv2(x)
       x = self.conv3(x)
       x = self.flatten(x)
       x = self.dense(x)
       return x

def create_spliceator_model(device='cuda'):
   model = Spliceator().to(device)
   optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed from Adamax to Adam
   criterion = nn.CrossEntropyLoss()
   
   return model, optimizer, criterion

def create_intsplice_model(device='cuda'):
   model = IntSplice().to(device)
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.CrossEntropyLoss()
   
   return model, optimizer, criterion

def create_deepsplicer_model(device='cuda'):
    model = DeepSplicer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion

def create_splicefinder_model(device='cuda'):
    model = SpliceFinder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion