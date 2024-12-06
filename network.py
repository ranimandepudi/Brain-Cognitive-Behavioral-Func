import torch.nn as nn
import torch.nn.functional as F
import torch

class Network(nn.Module):
    """
    aaasdasda 
    """
    def __init__(self,dropout, fc1_input_size, fc1_output_size, fc2_output_size):
        super().__init__()
        self.cv1 = nn.Conv3d(1,8, 3, stride=1, padding=0) #121*145*121
        self.bn1 = nn.BatchNorm3d(8)
        self.cv2 = nn.Conv3d(8, 16, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(16)
        self.cv3 = nn.Conv3d(16, 32, 3,stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(32)
        self.cv4 = nn.Conv3d(32, 64, 3,stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(64)
        self.cv5 = nn.Conv3d(64, 64, 3,stride=1, padding=0) 
        self.bn5 = nn.BatchNorm3d(64) 
        self.pool = nn.MaxPool3d(2)
        self.pool_large = nn.MaxPool3d(3)  # Larger pooling to reduce size

        # Fully connected layers
        self.fc1 = nn.Linear(fc1_input_size, fc1_output_size)
        self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)
     
        self.layer1 = nn.Sequential(self.cv1, self.bn1, nn.ReLU(), self.pool)
        self.layer2 = nn.Sequential(self.cv2, self.bn2, nn.ReLU(),self.pool)
        self.layer3 = nn.Sequential(self.cv3, self.bn3, nn.ReLU(),self.pool)
        self.layer4 = nn.Sequential(self.cv4, self.bn4, nn.ReLU(),self.pool)
        self.layer5 = nn.Sequential(self.cv5, self.bn5, nn.ReLU())


        self.convs = nn.Sequential(self.layer1,self.layer2,self.layer3,self.layer4,self.layer5)
        # self.convs.apply(Network.init_weights)
        self.dropout = nn.Dropout(dropout)
        print(f"Dropout layer: {self.dropout}")

   

    def forward(self, img,data=None):
       
        img = self.convs(img)
        if torch.isnan(img).any():
            print("NaN detected after convolutions")
        # img = self.global_pool(img)

        img = img.view(img.shape[0], -1)
        # print(f"Flattened size: {img.shape}")  # Debugging line to verify the size



        # Adding sex data to the convoluted data
        if data is not None:
            with torch.no_grad():
                img = torch.cat((img,torch.unsqueeze(data,1)),dim=1)
        
     
        # Passing through the fully connected layers with ReLU activation and dropout
        img = F.relu(self.fc1(img))
        self.fc1_output = img.cpu().detach().numpy()  # Store fc1 output
        img = self.dropout(img)
        img = self.fc2(img)
        self.fc2_output = img.cpu().detach().numpy()

        return img

    # @staticmethod
    # def init_weights(m):
    #     if isinstance(m, nn.Linear) or isinstance(m,nn.Conv3d):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         #m.bias.data.fill_(0.01)
       

