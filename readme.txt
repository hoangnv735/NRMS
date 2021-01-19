1. data structure

all articles.json
[
	{'id':1,'title':['this','is','title']},
	{'id':2,'title':['this','is','another','title']},
]

train_data.json / val_data.json / test_data.json
[
	{'user_id':1,'history':[1,2,3],'push':[4,5,6]},
	{'user_id':2,'history':[4,5,6],'push':[1,2,3]}
]

negative.pt
[1,2,3,4,5,6] # all item in 2 consecutive days

2. Train
run file main.py with appropriate arguments
eg: To train model with title segmented by pyvi
python3 main.py --epochs 10 --batch_size 50 --val_pos 1 --val_neg 4 --val_total 0 --embedding './word2vec/baomoi.window2.vn.model.bin' --train_data './data/all_data/train/users_list.json' --all_items './data/articles_with_pyvi/articles.json' --val_data './data/all_data/train/users_list.json' --val_items './data/all_data/negative/day18.pt' 

3. Test
run file test.py with appropriate arguments
eg: To test model on day 19, weight from file 'epoch1.pt'
python3 test.py --test_batch_size 50 --test_pos 2 --test_neg 8 --test_total 0 --load_model './checkpoint/epoch1.pt' --embedding './word2vec/baomoi.window2.vn.model.bin' --test_data './data/all_data/test/users_list_19.json' --all_items './data/articles_with_pyvi/articles.json'  --test_items './data/all_data/negative/day19.pt' 

