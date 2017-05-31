### Data Set- dataparser.py

## 1. create_data_label_path
   This function will create a dictionary which contains all mails cases and label of each cases
   #Input: Dataset path as list
   #Output: Path Dictionary - Key = case name
                              value = {'context':[mail context path list]
                                      'label': label path}

## 2. process_data_to_pickle
	This function will process raw data input pickle file
	#Input: dst_folder, 
			Path Dictionary //Derive from create_data_label_path

	#Output: Mail context pickle: list of words in the context
			 Label file pickle: Dict with each class in the lable files
