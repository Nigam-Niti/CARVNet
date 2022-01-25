from models import Bed_Network

bed_net = Bed_Network(sample_size = 112, sample_duration = 16, num_classes=65)
class SATA(nn.Module):
  def __init__(self):
      super(SATA, self).__init__()
      
      self.base_model = torch.nn.Sequential(*list(bed_net.children())[:-3])
      self.max_pool_t =  torch.nn.MaxPool3d(kernel_size=[2, 1, 1], stride=(1, 1, 1))
      self.max_pool_c =  torch.nn.MaxPool3d(kernel_size=[512, 1, 1], stride=(1, 1, 1))
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(8, 49)
      self.relu = nn.ReLU(inplace=True) 
      self.fc2 = nn.Linear(49, 256)
      self.fc3 = nn.Linear(50176, 512)
 

  def forward(self, x):
      print(x.shape)
      base = self.base_model(x)
      print("Bed net work",base.shape)

      # actor branch
      out = self.max_pool_t(base) 
      out = torch.permute(out,(0,2,1,3,4))
      out_tc = self.max_pool_c(out)
      out = self.dropout1(out_tc)
      out = torch.flatten(out, 1)
      out = self.relu(out)
      out = self.fc2(out)
      out = self.dropout2(out)
      sigmoid = torch.nn.Sigmoid()
      out1 = torch.nn.functional.softmax(out, dim=1)
      out2 = sigmoid(out)
      concat1 = torch.cat([out1, out2], dim=1)
      result = concat1.unsqueeze(1).unsqueeze(1).unsqueeze(1)
      out = torch.permute(result,(0,4,1,2,3))     
      recal_a = base + torch.mul(out,base)

      # theme branch
      one_mat = torch.ones(8,512).cuda()
      theme_score = one_mat - concat1
      theme_score =theme_score.unsqueeze(1).unsqueeze(1).unsqueeze(1)
      theme_score = torch.permute(theme_score,(0,4,1,2,3)) 
      out = torch.mul(base,theme_score)
      out = self.dropout1(out)
      out = torch.flatten(out, 1)
      out = self.relu(out)
      out = self.fc3(out)
      out = self.dropout2(out)
      sigmoid_t = torch.nn.Sigmoid()
      out = sigmoid_t(out)
      result_t = out.unsqueeze(1).unsqueeze(1).unsqueeze(1)
      out = torch.permute(result_t,(0,4,1,2,3))     
      recal_t = base + torch.mul(out,base)  
      
      return recal_a, recal_t
