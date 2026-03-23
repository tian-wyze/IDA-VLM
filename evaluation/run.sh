#!/bin/bash

python eval_internvl.py ../prepare_dataset/household-based/family_crossclothes_samecamera.json ../../tian_data/wyze_person_v2/cross_clothes/
python eval_internvl.py ../prepare_dataset/household-based/family_crossclothes_crosscamera.json ../../tian_data/wyze_person_v2/cross_clothes/
python eval_internvl.py ../prepare_dataset/household-based/singleton_crossclothes_samecamera.json ../../tian_data/wyze_person_v2/cross_clothes/
python eval_internvl.py ../prepare_dataset/household-based/singleton_crossclothes_crosscamera.json ../../tian_data/wyze_person_v2/cross_clothes/

