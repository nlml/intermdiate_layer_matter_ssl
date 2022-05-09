import numpy as np
import os

os.chdir("/workspace/diabetic_retinopathy/data/eyepacs/bin2/train")

valid_p = 0.2

rng = np.random.RandomState(1)

for c in [0, 1]:
    c = str(c)
    ld = os.listdir(c)
    ids = set([i.split("_")[0] for i in ld])
    n_valid = int(len(ids) * valid_p)
    ids = np.array(list(ids))[rng.permutation(len(ids))]
    valid_ids = ids[:n_valid]
    train_ids = ids[n_valid:]

    valid_imgs = []
    for i in valid_ids:
        valid_imgs += [j for j in ld if j.split("_")[0] == i]
    vi = set(valid_imgs)
    train_imgs = [i for i in ld if i not in vi]
    print(c, len(valid_imgs), len(train_imgs))
    os.makedirs(f"../validation/{c}", exist_ok=True)
    print(c)
    for i in valid_imgs:
        os.rename(f"{c}/{i}", f"../validation/{c}/{i}")
        print(f"{c}/{i}", f"../validation/{c}/{i}")
    print(c)
