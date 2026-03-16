import os, yaml, torch
from PIL import Image
import torchvision.transforms as T
from models.generator_mobile import GeneratorMobile

@torch.no_grad()
def run(weights, img_path, out_path='enhanced.png', size=256, device='cuda'):
    tf = T.Compose([T.Resize((size,size)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    x = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
    G = GeneratorMobile().to(device).eval()
    if os.path.isfile(weights): G.load_state_dict(torch.load(weights, map_location=device))
    y = G(x).cpu().squeeze(0); y = (y.clamp(-1,1)*0.5+0.5)
    T.ToPILImage()(y).save(out_path); print('saved to', out_path)

if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='outputs/checkpoints/G_ep50.pth')
    ap.add_argument('--img', required=True)
    ap.add_argument('--out', default='enhanced.png')
    ap.add_argument('--size', default=256, type=int)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    run(args.weights, args.img, args.out, args.size, args.device)
