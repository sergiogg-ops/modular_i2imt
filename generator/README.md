# Automatic Sinthetic I2IMT Dataset Generator
Utilities to create sinthetic I2IMT datasets from MT and image datasets with the `data_generator` script. The hardness of the resulting dataset can be tooggled by specifying a variable interval of font sizes, different fonts and varing slopes in the text direction. This script can also be fed with a  configuration file so the parameters for its creation can be saved. Furthermore, a seed can be delivered so the creation of the dataset can be reproduced later.

The `images.py` script can be used to create a set of images with a plain background of random color.

## Available fonts
- [Arial](https://github.com/sergiogg-ops/I2IMT/tree/main/generator/font/Arial)
- [OpenSans](https://github.com/sergiogg-ops/I2IMT/tree/main/generator/font/OpenSans)