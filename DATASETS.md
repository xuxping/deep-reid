# How to prepare data

Create a directory to store reid datasets under this repo via
```bash
cd deep-reid/
mkdir data/
```


## Image ReID

**Market1501**:
1. Download the dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html.
2. Extract the file and rename it to `market1501`. The data structure should look like:
```
market1501/
    bounding_box_test/
    bounding_box_train/
    ...
```
3. Use `market1501` as the key to load Market1501.

