import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams["font.family"]="serif"; plt.rcParams["font.serif"]=["Liberation Serif","Times New Roman","DejaVu Serif"]
files=[("/mnt/user-data/uploads/927e7c2e-image.png","(a) Base Stable Diffusion 1.5\n(no adapter)"),
       ("/mnt/user-data/uploads/ca45462e-image.png","(b) Conditional WGAN-GP\n(final checkpoint)"),
       ("/mnt/user-data/uploads/98680559-image.png","(c) Stable Diffusion 1.5 + LoRA\n(final checkpoint)")]
rows=["AKIEC","BCC","DF","MEL","VASC"]
fig,axes=plt.subplots(1,3,figsize=(6.6,3.75))
for ax,(f,t) in zip(axes,files):
    im=Image.open(f).convert("RGB"); assert im.size==(512,640), im.size
    ax.imshow(im,interpolation="nearest"); ax.set_title(t,fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([64+128*i for i in range(5)])
    ax.set_yticklabels(rows if ax is axes[0] else [],fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_linewidth(0.6)
plt.tight_layout(w_pad=0.6)
plt.savefig("Figure_5_3_1_Generator_Inspection.png",dpi=300,bbox_inches="tight")
im=Image.open("Figure_5_3_1_Generator_Inspection.png"); print(im.size, im.info.get("dpi"))
