import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def adain(content_feat, style_feat, eps=1e-5):
    c_mean, c_std = content_feat.mean([2,3], keepdim=True), content_feat.std([2,3], keepdim=True) + eps
    s_mean, s_std = style_feat.mean([2,3], keepdim=True), style_feat.std([2,3], keepdim=True) + eps
    return s_std * ((content_feat - c_mean) / c_std) + s_mean

def load_image(path, max_dim=512):
    img = Image.open(path).convert("RGB")
    scale = max_dim / max(img.size)
    img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)

def display_image(tensor, title=""):
    import matplotlib.pyplot as plt
    img = tensor.detach().cpu().squeeze(0).clamp(0,1)
    plt.imshow(img.permute(1,2,0))
    plt.title(title); plt.axis("off"); plt.show()

def gram_matrix(x):
    b, c, h, w = x.size()
    f = x.view(b, c, h*w)
    G = torch.bmm(f, f.transpose(1,2))
    return G / (c * h * w)

def total_variation_loss(x):
    return (torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
          + torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum())
