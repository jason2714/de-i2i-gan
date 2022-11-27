# de-i2i-gan
data efficient GAN for image-to-image translation

# architecture
The major architecture of G mainly follows the commonly used image-to-image translation networks, which first encodes the input image by a stride of 4, and then decodes it to its original size.

(pix2pix, cycle-GAN, star-GAN)    
![pix2pix-G.png](images/pixG.png)
![cycleGAN-G.png](images/cycleG.png)
![starGAN-G.png](images/starG.png)
![cycleG2.png](https://miro.medium.com/max/4800/1*IPAXkl4Cz2RsDwHPHck7Xw.png)
[Style-Gan github](https://github.com/yunjey/stargan)  
To improve synthesis realism and diversity for defect generation, we specifically design spatial and categorical control, stochastic variation and layer-based composition for G.
```angular2html
My thoughts
base on star-GAN
                  SPADE .. ..  ..  ..
7k1s 4k2s 4k2s  3k1s..... 4k1su 4k1su 3k1s
3-64-128-256-256-256-256-256-256-256-128-64-3
  |   |   |   |   |       |   |   |   |   | 
  |   |   |   |   |-skip--|   |   |   |   | 
  |   |   |   |-----skip------|   |   |   | 
  |   |   |---------skip----------|   |   | 
  |   |-------------skip--------------|   | 
  |-----------------skip------------------|

use involution instead of SPADE to aggregate spatial map
CC-FPSE:
use FPN to get global information and spatial agnostic depth-wise convolution
use involution instead and use inceptionV3 to simulate H(x)

adding noise  
http://proceedings.mlr.press/v139/feng21g/feng21g.pdf
noise has two type
x = feature + noise * r
noise = (B, 1, H, W)
1. r = (1, C, 1, 1)
2. r = (1, 1, 1, 1)

1. add skip connection
2. maybe add SPADE
3. use leaky relu with 0.2 in encoder, relu in decoder
4. instanceNorm instead of BatchNorm
5. TTUR
6. SpectralNorm Of course
7. use res-blk in spade
```
The network architecture of D is the same as StarGAN, which includes a D src to distinguish faked samples from real ones using PatchGAN and a D cls to predict the categories of generated defect.
![starGAN-D.png](images/starD.png)
(star-GAN)

i.e.,Ais acquired by spatial-wisely repeating the target de-fect labelc∈RC. This restriction can be lifted duringinference stage, which enables Defect-GAN to add defectsat different location(s) in a context-compatible mann

# loss
```angular2html
50,000 normal image patches
Images are re-sized to 128×128 for training
one generator update every five discriminator update
Adam optimizer with β1=0.5 and β2=0.999
learning rate starting from 2×10−4 and reducing to 1×10−6
We set batch size at 4 and the total training iteration at 500,000


GAN-loss  
(Wasserstein GAN objective with gradient penalty)
cls-loss  
reconstruction-loss  
spatial distribution loss  
(md <-> md~), (mn <-> mn~)  
spatial distribution L1-loss  
(md,mn <-> 0)  
LD = gan + a*cls  
LG = gan + b*cls + c*rec + d*sd + e*sdl1  
(a, b, c, d, e) -> (2, 5, 5, 5, 1)  
```