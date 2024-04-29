import csv
from PIL import Image
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
from os.path import abspath, dirname
from tqdm import tqdm
import argparse

metrics_path = os.path.join(
    dirname(dirname(abspath(__file__))),
    'tasks',
    'utils',
    'metrics'
)
import sys
sys.path.append(metrics_path)
from id_score import calculate_id_score
from object_score import calculate_object_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept_type', type=str, default='id')
    parser.add_argument('--concept', type=str, default='angelina jolie')
    parser.add_argument('--threshold', type=float, default=0.99)
    args = parser.parse_args()

    concept_type = args.concept_type # 'id', 'object'
    concept = args.concept # the concept to restore
    threshold = args.threshold # we use 0.97 for 'jeep', 0.99 for 'angelina jolie'

    data_fold = os.path.join(
        dirname(dirname(dirname(abspath(__file__)))),
        'files',
        'dataset',
        concept_type,
        concept.split()[-1]
    )
    image_fold = os.path.join(data_fold, 'imgs')
    choose_ls = []

    if concept_type == 'id':
        img_ls = os.listdir(image_fold)
        img_ls = sorted(img_ls, key=lambda x: int(x.split('_')[0]))
        # print(img_ls)

        for i in tqdm(img_ls):           
            image = Image.open(os.path.join(image_fold, i))
            score = calculate_id_score(image, concept)
            if score > threshold:
                choose_ls.append([int(i[:-6]), score])

    elif concept_type == 'object':
        img_ls = os.listdir(image_fold)
        img_ls = sorted(img_ls, key=lambda x: int(x.split('_')[0]))

        for i in tqdm(img_ls): 
            image_path = os.path.join(image_fold, i)
            score = calculate_object_score(image_path, concept)
            if score > threshold:
                choose_ls.append([int(i[:-6]), score])

    choose_ls.insert(0, ['case_number', 'score'])
    csv_file_path = os.path.join(data_fold, 'choose.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in choose_ls:
            writer.writerow(row)
    print("Finish choosing good training images.")