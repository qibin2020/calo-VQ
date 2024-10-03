Implementation of Calo-VQ [2405.06605]
=======
# Environment
Current scripts need `pytorch-lightning==1.6.5`. For more check `environment.yaml`

# Training
```
python main.py --base config/xxx_step1.yaml -t True --gpus 1 -l logs/xxx_step1 -n xxx_step1
python main.py --base config/xxx_step2.yaml -t True --gpus 1 -l logs/xxx_step2 -n xxx_step2
```

# Generation
```
python gen-tools.py --out zzz.h5 --model models/xxx_step2 --type yyy
```

# Thanks to
We would like to express our gratitude to the authors of the codes used in this work: [taming-transformers](https://github.com/CompVis/taming-transformers) and [minGPT](https://github.com/karpathy/minGPT) 
