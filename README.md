## Consistent Dialogue Generation with Self-supervised Feature Learning (CoCon)

This repo contains the implementation of the paper ["Consistent Dialogue Generation with Self-supervised Feature Learning"](https://arxiv.org/abs/1903.05759)


## Environment

Tensorflow == 1.5.1

## Run

Training CoCon Discriminator

``python feature_extractor.py`` for feature extraction.

Training CoCon Generator

```
python cocon.py -global # using global/topic feature to training a CoCon-T
python cocon.py -global -local # using both global/topic and local/persona feature to training a CoCon-TP
```
Other option information can be found in the help `python cocon.py --help`. 


Generating controlled response by modifying bits:

```
python interpolation.py -global -local --feed -tf /newdata2/test_old.txt --bit 25
```


## Cite
Our paper can be cited at

```
@article{zhang2019consistent,
  title={Consistent Dialogue Generation with Self-supervised Feature Learning},
  author={Zhang, Yizhe and Gao, Xiang and Lee, Sungjin and Brockett, Chris and Galley, Michel and Gao, Jianfeng and Dolan, Bill},
  journal={arXiv preprint arXiv:1903.05759},
  year={2019}
}
```


