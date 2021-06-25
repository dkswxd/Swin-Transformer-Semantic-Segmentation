import cv2
import numpy as np
import torch.nn.functional as F

# attn2 = F.softmax(attn.view(2560,3,64*64) - attn.mean(dim=(2,3)).view(2560,3,1),dim=-1).view(2560,3,64,64)
attn2 = attn - attn.mean(dim=(-1),keepdim=True)
attn2 = attn2 / attn2.std(dim=(-1),keepdim=True)
attn_windows = attn2[:,0,:,:].sum(dim=-2).view(-1, 4, 4, 4, 1)
# attn_windows = attn[:,0,:,:].sum(dim=-2).view(-1, 4, 4, 4, 1)
attn_map = dilate_window_reverse(attn_windows, (4,4,4),(1,1,1),8,128,160)
attn_map = attn_map[0,:,:,:,0].detach().cpu().numpy()
attn_map -= attn_map.min()
attn_map /= attn_map.max()
attn_map *= 255
attn_map = attn_map.astype(np.uint8)
for i in range(attn_map.shape[0]):
    cv2.imwrite(f'{i}.png', cv2.resize(attn_map[i,:,:], (640, 512)))




# attn2 = attn - attn.mean(dim=(-1),keepdim=True)
# attn2 = attn2 / attn2.std(dim=(-1),keepdim=True)
# attn_windows = attn2[:,0,:,:].sum(dim=-1).view(-1, 7, 7, 1)
attn_windows = attn[:,0,:,:].sum(dim=-2).view(-1, 7, 7, 1)
attn_map = window_reverse(attn_windows, 7,133,161)
attn_map = attn_map[0,:,:,0].detach().cpu().numpy()
attn_map -= attn_map.min()
attn_map /= attn_map.max()
attn_map *= 255
attn_map = attn_map.astype(np.uint8)
cv2.imwrite('00.png', cv2.resize(attn_map[:,:], (161*8, 133*8)))