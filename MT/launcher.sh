#!/bin/bash

python nllb.py ../data/horizontal_mix.de -src de -tgt en --output predictions/en/nllb.en
python madlad.py ../data/horizontal_mix.de -src de -tgt en --output predictions/en/madlad.en
python seed.py ../data/horizontal_mix.de -src de -tgt en --output predictions/en/seed.en
python gemma.py ../data/horizontal_mix.de -src de -tgt en --output predictions/en/gemma.en

python nllb.py ../data/horizontal_mix.en -src en -tgt de --output predictions/de/nllb.de
python madlad.py ../data/horizontal_mix.en -src en -tgt de --output predictions/de/madlad.de
python seed.py ../data/horizontal_mix.en -src en -tgt de --output predictions/de/seed.de
python gemma.py ../data/horizontal_mix.en -src en -tgt de --output predictions/de/gemma.de