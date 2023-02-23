# FI-DUSGAN
This is the official implementation for [Video Frame Interpolation Via Down-Up Scale Generative Adversarial Networks (2022)](https://doi.org/10.1016/j.cviu.2022.103434).

# Requirements
For using this implementation, we recommend PyTorch with version 1.5.0 or later.

# Dataset
The target dataset of this implementation is the Vimeo-90k dataset. The sample data directory is organized following the frame interpolation subset of the Vimeo90k.

Please see the Vimeo-90k dataset documentation for more details.

# Run the code
```python train.py --path=cpt_folder```

# Pre-trained model
Please file it on [Google Drive](https://drive.google.com/file/d/1VsVX13DKYpwNhxii-mSyAqlzeAlg_sEO/view?usp=sharing) and then put in the pre-trained folder. Note that rename the file to "net_gen.pt" might be required.

# Reference

```
@ARTICLE{9097443,
  author={Tran, Quang Nhat and Yang, Shih-Hsuan},
  journal={Computer Vision and Image Understanding}, 
  title={Video Frame Interpolation Via Down-Up Scale Generative Adversarial Networks}, 
  year={2022},
  volume={220},
  doi={https://doi.org/10.1016/j.cviu.2022.103434}}
```
