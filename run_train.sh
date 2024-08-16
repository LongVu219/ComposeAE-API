bert-serving-start -model_dir bert-model/bert_uncase -num_worker=2

python -W ignore  main.py --dataset=fashion200k --dataset_path=data/fashion200k  --model=composeAE --loss=batch_based_classification --learning_rate_decay_frequency=50000 --num_iters=70000 --use_bert True --use_complete_text_query True --weight_decay=5e-5 --comment=fashion200k_composeAE > /home/longvv/ComposeAE/logs/logs1.txt  

python -W ignore  test.py --dataset=fashion200k --dataset_path=data/fashion200k  --model=composeAE --loss=batch_based_classification --learning_rate_decay_frequency=50000 --num_iters=7 --use_bert True --use_complete_text_query True --weight_decay=5e-5 --comment=fashion200k_composeAE 

python -W ignore  inference.py --dataset=fashion200k --dataset_path=data/fashion200k  --model=composeAE --loss=batch_based_classification --learning_rate_decay_frequency=50000 --num_iters=7 --use_bert True --use_complete_text_query True --weight_decay=5e-5 --comment=fashion200k_composeAE 

python -W ignore  check_model.py --dataset=fashion200k --dataset_path=data/fashion200k  --model=composeAE --loss=batch_based_classification --learning_rate_decay_frequency=50000 --num_iters=7 --use_bert True --use_complete_text_query True --weight_decay=5e-5 --comment=fashion200k_composeAE
