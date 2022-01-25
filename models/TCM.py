from models import SATA

class TCM(nn.Module):
  def __init__(self):
      super(TCM, self).__init__()
      self.sata = SATA()
      #self.base_model = torch.nn.Sequential(*list(sata.children())[0:])
      self.avg_pool_t =  torch.nn.AvgPool3d(kernel_size=[1, 7, 7], stride=(1, 1, 1))
      self.avg_pool_c =  torch.nn.AvgPool3d(kernel_size=[512, 1, 1], stride=(1, 1, 1))
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(8, 2)
      self.relu = nn.ReLU(inplace=True) 
      self.fc2 = nn.Linear(2, 512)
      self.dila1_conv = torch.nn.Conv3d(512, 512, kernel_size=(1,1,1), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
      self.dila2_conv = torch.nn.Conv3d(512, 512, kernel_size=(1,1,1), stride=1, padding=0, dilation=2, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
      self.dila3_conv = torch.nn.Conv3d(512, 512, kernel_size=(1,1,1), stride=1, padding=0, dilation=3, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
      self.dila4_conv = torch.nn.Conv3d(512, 512, kernel_size=(1,1,1), stride=1, padding=0, dilation=4, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)       
   
  def forward(self, x, y):
      T_a, T_m = self.sata(x)
      # actor branch
      out = self.avg_pool_t(T_a) 
      out = torch.permute(out,(0,2,1,3,4))
      out_tc = self.avg_pool_c(out)
      out = self.dropout1(out_tc)
      out = torch.flatten(out, 1)
      out = self.relu(out)
      out = self.fc2(out)
      out = self.dropout2(out)
      sigmoid = torch.nn.Sigmoid()
      out2 = sigmoid(out)
      result_t = out2.unsqueeze(1).unsqueeze(1).unsqueeze(1)  
      print(result_t.shape) 
      out = torch.permute(result_t,(0,4,1,2,3))
      recal_a = T_a + torch.mul(out,T_a)
      stack1 = self.dila1_conv(recal_a)
      stack2 = self.dila2_conv(recal_a)
      stack3 = self.dila3_conv(recal_a)
      stack4 = self.dila4_conv(recal_a)
      stack_a = stack1 + stack2 + stack3 + stack4
     
       # theme branch
      stack11 = self.dila1_conv(T_m)
      stack21 = self.dila2_conv(T_m)
      stack31 = self.dila3_conv(T_m)
      stack41 = self.dila4_conv(T_m)
      stack_a1 = stack11 + stack21 + stack31 + stack41

      #concat
      concat1 = stack_a + stack_a1

     
      return concat1
