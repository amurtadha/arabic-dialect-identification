# arabic-dialect-identification


 A model for arabic-dialect-identification.
 
 The source code for the paper: ABDELMAJEED Mohammed and  Murtadha, Ahmed, et al. "A Three-Stage Neural Model for  Arabic Dialect Identification" [[1]](). 
 

 

# Data

* The datasets used in our experminents can be downloaded from  [MADAR](https://docs.google.com/forms/d/e/1FAIpQLSe0PRkK8uetqWTlxGAUn7MupNcRO3HOHQeHK4_xkoZ7TAh98g/viewform), [QADI](https://github.com/qcri/QADI) and [NADI](https://github.com/UBC-NLP/nadi)
* We have processed Nadi dataset datasets/Nadi/
* The embeddings of Nadi dataset for both cbow58 and arabvec [download](https://drive.google.com/drive/folders/1RY2-_gW5jXRtSBn4yf_HuDPRPZVV2qjN?usp=sharing) and put them to embedding/
* To build the embeddings and tokenizer of the other datasets:
	* download these models:
    		* [cbow58](https://drive.google.com/open?id=0B2WzDD9FC2KXRHlYNjYxUmowRW8)
    		* [arabvec](https://github.com/bakrianoo/aravec)
    
	* unzip them into embedding/



# Prerequisites:
Required packages are listed in the requirements.txt file:

```
pip install -r requirements.txt
```
# Pre-processing

* The data should be in the following format:
```
[
  {
      "text": "الفار العور يشوف فقط كيسي ومايشوف ماتويد",
      "label": "Iraq"
   }
]
```


# Training: 
* To train  Bert-based ADI: 
	* Run the following code 
	```
	python train.py --dataset Nadi --baseline camel --device_group 0
	```
	* The params could be :
    		
		- --baseline ={camel,arbert, mdb}
		- --dataset =\{Nadi,QADI,Corpus-6, Corpus-9, Corpus-26\}	
		- --device_group ="0" # GPU ID

The the results will be written into results_bert.txt
* To train  Word2vec-based ADI: 
```
python train_LSTM.py --dataset Nadi --embeding cbow58 --device_group 0
```
  * The params could be :
    	
	- --embeding=\{cbow58, arabvec\}
	- --dataset =\{Nadi,QADI,Corpus-6, Corpus-9, Corpus-26\}	
	- --device_group ="0" # GPU ID

The the results will be written into results_lstm.txt

If you use the code,  please cite the paper: 
```

```
