# download models
mkdir models
curl https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv -o all_w100.tsv
curl https://nlp.cs.washington.edu/xorqa/cora/models/mGEN_model.zip -o mGEN_model
curl https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt -o mDPR_biencoder_best
unzip mGEN_model.zip
mkdir embeddings
cd embeddings

# for x in {0..7}; do curl -O https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_en_{$x}; done; 
for i in 0 1 2 3 4 5 6 7;
do 
  curl https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_en_$i 
done

# for x in {0..7}; do curl -O https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_others_{$x}; done; 
for i in 0 1 2 3 4 5 6 7;
do 
  curl https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_others_$i  
done
cd ../..

# download eval data
mkdir data
cd data
curl -O https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
curl -O https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_retrieve_eng_span.jsonl
curl -O https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_full.jsonl
cd ..

# Run mDPR
cd mDPR
python dense_retriever.py \
    --model_file ../models/mDPR_biencoder_best.cpt \
    --ctx_file ../models/all_w100.tsv \
    --qa_file ../data/xor_dev_full_v1_1.jsonl \
    --encoded_ctx_file "../models/embeddings/wikipedia_split/wiki_emb_*" \
    --out_file xor_dev_dpr_retrieval_results.json \
    --n-docs 20 --validation_workers 1 --batch_size 256 --add_lang
cd ..

# modified run
python dense_retriever.py \
    --model_file /Users/johncmcc/Projects/NLP/CORA-Analysis/models/mDPR_biencoder_best.cpt \
    --ctx_file ../models/all_w100.tsv \
    --qa_file /Users/johncmcc/Projects/NLP/CORA-Analysis/data/xor_dev_full_v1_1.jsonl \
    --encoded_ctx_file "../models/embeddings/wikipedia_split/wiki_emb_*" \
    --out_file xor_dev_dpr_retrieval_results.json \
    --n-docs 20 --validation_workers 1 --batch_size 256 --add_lang
cd ..

# Convert data 
cd mGEN
python3 convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp ../mDPR/xor_dev_dpr_retrieval_results.json \
    --output_dir xorqa_dev_final_retriever_results \
    --top_n 15 \
    --add_lang \
    --xor_engspan_train ../data/xor_train_retrieve_eng_span.jsonl \
    --xor_full_train ../data/xor_train_full.jsonl \
    --xor_full_dev ../data/xor_dev_full_v1_1.jsonl

# Run mGEN
CUDA_VISIBLE_DEVICES=0 python eval_mgen.py \
    --model_name_or_path \
    --evaluation_set xorqa_dev_final_retriever_results/val.source \
    --gold_data_path xorqa_dev_final_retriever_results/gold_para_qa_data_dev.tsv \
    --predictions_path xor_dev_final_results.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 4
cd ..

# Run evaluation
cd eval_scripts
python eval_xor_full.py --data_file ../data/xor_dev_full_v1_1.jsonl --pred_file ../mGEN/xor_dev_final_results.txt --txt_file