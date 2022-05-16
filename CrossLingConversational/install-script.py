""" Script for downloading models.


You can either use the version hosted by the SentEval team, which is already tokenized,
or you can download the original data from (https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi) and extract the data from it manually.
For Windows users, you can run the .msi file. For Mac and Linux users, consider an external library such as 'cabextract' (see below for an example).
You should then rename and place specific files in a folder (see below for an example).
mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi

"""

import argparse
import os
import sys
from urllib import request, parse
import zipfile
import progressbar


#TASKS = ["all_w100", "mGEN_Model", "mDPR_Biencoder", "wiki_emb", "wiki_emb_others", "Eval_data_1", "Eval_data_2", "Eval_data_3"]
TASKS = ["wiki_emb_", "wiki_emb_others_", "Eval_data_1", "Eval_data_2", "Eval_data_3"]
TASK2PATH = {
    #"all_w100": "https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv",
    #"mGEN_Model": "https://nlp.cs.washington.edu/xorqa/cora/models/mGEN_model.zip",
    #"mDPR_Biencoder": "https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt",
    "wiki_emb_": "https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_en_",
    "wiki_emb_others_": "https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_others_",
    "Eval_data_1": "https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl",
    "Eval_data_2": "https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_retrieve_eng_span.jsonl",
    "Eval_data_3": "https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_full.jsonl"
}


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

"""def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    ext = os.path.splitext(TASK2PATH[task])
    data_file = "%s.%s" % (task, ext)

    if not os.path.isdir(os.path.join(data_dir, "models")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")"""


def download(task, data_dir, folder):
    print("Downloading %s..." % task)
    ext = os.path.splitext(TASK2PATH[task])
    
    if not os.path.isdir(os.path.join(data_dir, folder)):
        os.mkdir(os.path.join(data_dir, folder))
    file = "%s%s" % (task, ext[1])
    data_file = os.path.join(data_dir, folder, file)    
    request.urlretrieve(TASK2PATH[task], data_file, MyProgressBar())
    print("\tCompleted!")
    return

def download_embeddings(task, data_dir):
    for i in range(1,7):
        print("Downloading %s%s..." % (task, i))
        url = TASK2PATH[task] + str(i)
        
        if not os.path.isdir(os.path.join(data_dir, "embeddings")):
            os.mkdir(os.path.join(data_dir, "embeddings"))
        
        file = "%s%s" % (task, i)
        data_file = os.path.join(data_dir, "embeddings", file)
        request.urlretrieve(url, data_file, MyProgressBar())
        print("\tCompleted!")
    return


def get_tasks(task_names):
    task_names = task_names.split(",")
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="directory to save data to", type=str, default="downloads")
    parser.add_argument(
        "--tasks", help="tasks to download data for as a comma separated string", type=str, default="all"
    )
    parser.add_argument(
        "--path_to_mrpc",
        help="path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt",
        type=str,
        default="",
    )
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task.__contains__("Eval"):
            download(task, args.data_dir, "data")
        elif task.__contains__("emb"):
            download_embeddings(task, args.data_dir)
        else:
            download(task, args.data_dir, "models")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
