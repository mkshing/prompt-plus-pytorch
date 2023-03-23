`cat_statue` and `mug_skulls` are taken from the [original Textual Inversion repository](https://github.com/rinongal/textual_inversion#pretrained-models--data) 
and resized to 512x512 by the following code.
```python
import os
from PIL import Image

image_dir = "image-path"
for file_path in os.listdir(image_dir):
    image_path = os.path.join(save_path, file_path)
    Image.open(image_path).resize((512, 512)).save(image_path)
```