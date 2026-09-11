conda install -c nvidia cudnn=9.8

pip install pillow clu tensorflow==2.15.0 "keras<3" "torch<=2.4" torchvision tensorflow_datasets matplotlib==3.9.2
pip install orbax-checkpoint==0.4.4 ml-dtypes==0.5.0 tensorstore==0.1.67
pip install "diffusers==0.30.3" "protobuf<5" dm-tree cached_property   # diffusers>=0.33 dropped Flax; protobuf 5 breaks tensorflow 2.15

# test with jax and jaxlib 0.6.2
pip install -U "jax[cuda12]" 