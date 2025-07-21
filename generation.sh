python gen-tools.py --out ds1pion.h5 --model models/ds1pion-model-final/2023-08-09T00-34-37_2_2_64_3h --type 1_pion --batch-size 100 # for ds1, cond from predefined sequence so fixed event number.
python gen-tools.py --out ds1photon.h5 --model models/ds1photon-model-final/2023-08-10T01-45-28_2_2_64_3h --type 1_photon --batch-size 100 # for ds1, cond from predefined sequence so fixed event number.
python gen-tools.py --out ds2.h5 --model models/ds2-model-final_new/2023-08-10T16-46-18_2_2_64_5h_r1 --type 2 --nevts 100000  --batch-size 100
python gen-tools.py --out ds3.h5 --model models/ds3-model-final/2023-05-29T05-37-00_gpt_1_1_16 --type 3 --nevts 100000  --batch-size 100
python gen-tools.py --out ds3_norm.h5 --model models/ds3-model-final-slow/2023-08-18T23-29-56_1_1_128 --type 3 --nevts 100000  --batch-size 100
echo DONE ALL