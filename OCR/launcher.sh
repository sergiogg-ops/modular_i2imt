#!/bin/bash

python easy-ocr.py /home/sgomgon/modular_i2imt/data/horizontal_mix_en/* -l en --output predictions/en/easyocr.en -gpu
python trocr.py /home/sgomgon/modular_i2imt/data/horizontal_mix_en/* --output predictions/en/trocr.en -gpu
python deepseek.py /home/sgomgon/modular_i2imt/data/horizontal_mix_en/* --output predictions/en/deepseek.en -gpu

python easy-ocr.py /home/sgomgon/modular_i2imt/data/horizontal_mix_de/* -l de --output predictions/de/easyocr.de -gpu
python trocr.py /home/sgomgon/modular_i2imt/data/horizontal_mix_de/* --output predictions/de/trocr.de -gpu
python deepseek.py /home/sgomgon/modular_i2imt/data/horizontal_mix_de/* --output predictions/de/deep -gpu