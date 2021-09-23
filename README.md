The code is based on the [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline).

### Prerequisites

- python >= 0.4 
- torchvision
- ignite=0.1.2
- ycas
### Dataset
You could access all the datasets with this link [BaiduYun](https://pan.baidu.com/s/1tOq9I84YY4CWbh7hlw8NIA) [hf31].

(1) Market1501
* Extract dataset and rename to `market1501`. The data structure would like:
```bash
data
    market1501 # this folder contains 6 files.
        bounding_box_test/
        bounding_box_train/
        query/
        ......
```
(2) DukeMTMC-reID

   * Extract dataset and rename to `dukemtmc-reid`. The data structure would like:

   
    data
        dukemtmc-reid
        	DukeMTMC-reID 
            	bounding_box_test/
            	bounding_box_train/
            	query/
            	......
 
 (3) DukeMTMC-VideoReID
     * Extract dataset and rename to `DukeMTMC-VideoReID`. The data structure would like:

   
    data
        DukeMTMC-VideoReID 
            train/
            query/
            gallery/
            ......
            	
 (4) MARS
  * Extract dataset and rename to `MARS`. The data structure would like:

   
    data
        MARS
            bbox_test/
            bbox_train/
            info/
            ......
            	
  (5) MSMT17
    * Extract dataset and rename to `msmt17`. The data structure would like:
    
    data
        msmt17
            bounding_box_test/
            bounding_box_train/
            query/
            ......

            	
###  Pretrained Models

Our pretrained models could be obtained in [BaiduYun](https://pan.baidu.com/s/1YeB5AhHaH8Jym6qPGBeOZg) [d0d0].


### Usage  
To test model, please run   
    
    python tools/test.py --config_file='configs/test.yml' MODEL.DEVICE_ID "('0')" 
    DATASETS.NAMES "('market1501|dukemtmc|dukemtmc-video|mars|msmt17')" DATASETS.ROOT_DIR "('data')"
    MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('[your model path]')"
    


 

