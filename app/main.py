from model import StyleEncoder, StyleDecoder
from utils import load_image, display_image, gram_matrix, total_variation_loss
import torch
import torch.nn.functional as F
import torch.optim as optim
from piq import psnr, ssim
import lpips
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_decoder(content_path, style_path,
                  steps=1000,
                  style_weight=1e5,
                  content_weight=1.0,
                  tv_weight=1e-5,
                  lr=1e-3):
    content_img = load_image(content_path)
    style_img   = load_image(style_path)

    encoder = StyleEncoder().to(device).eval()
    decoder = StyleDecoder().to(device).train()

    with torch.no_grad():
        content_feats = encoder(content_img)
        style_feats   = encoder(style_img)

    optimizer = optim.Adam(decoder.parameters(), lr=lr)

    for step in range(1, steps+1):
        optimizer.zero_grad()
        output = decoder(content_feats, style_feats, alpha=0.6)
        feats_out = encoder(output)

        # Content loss
        enc4_out = F.interpolate(feats_out["enc4"], size=content_feats["enc4"].shape[-2:], mode="bilinear", align_corners=False)
        c_loss = F.mse_loss(enc4_out, content_feats["enc4"]) * content_weight

        # Style loss
        s_loss = 0
        style_ws = {"enc1":0.5, "enc2":1.0, "enc3":1.5, "enc4":3.0, "enc5":4.0}
        for layer, w in style_ws.items():
            f_out = F.interpolate(feats_out[layer], size=style_feats[layer].shape[-2:], mode="bilinear", align_corners=False)
            gm_o = gram_matrix(f_out)
            gm_s = gram_matrix(style_feats[layer])
            s_loss += w * F.mse_loss(gm_o, gm_s)
        s_loss *= style_weight

        tv_loss = tv_weight * total_variation_loss(output)
        loss = c_loss + s_loss + tv_loss
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}/{steps} — Loss: {loss.item():.2f}")

    print("Training done.")

    # Evaluation
    final_output = output.detach().clamp(0, 1)
    content_img_resized = F.interpolate(content_img, size=final_output.shape[-2:], mode="bilinear", align_corners=False)
    content_img_resized = content_img_resized.detach().clamp(0, 1)

    print(f"PSNR: {psnr(final_output, content_img_resized, data_range=1.0).item():.2f}")
    print(f"SSIM: {ssim(final_output, content_img_resized, data_range=1.0).item():.4f}")
    lpips_score = lpips.LPIPS(net='alex').to(device)(final_output, content_img_resized)
    print(f"LPIPS: {lpips_score.item():.4f}")

    # Save output
    output_img = final_output.cpu().squeeze(0).permute(1, 2, 0).numpy()
    Image.fromarray((output_img * 255).astype("uint8")).save("output.jpg")
    print("Output image saved as output.jpg")

if __name__ == "__main__":
    train_decoder("input.jpg", "style_transfer.jpg")
