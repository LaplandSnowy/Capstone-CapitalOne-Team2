# -*- coding: utf-8 -*-
"""
Parse xml data to csv file

Examples
--------
python parse_xml.py -i ABSA16_Restaurants_Train_SB1_v2.xml
"""

import os
import sys
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd

sys.path.append(os.path.abspath('..'))


def parse_args(arguments):
    argparser = ArgumentParser()
    argparser.add_argument(
        '-i',
        '--input',
        default="Restaurants/sub_task_1/ABSA16_Restaurants_Train_SB1_v2.xml",
        help='The path to the xml dat file for parsing')
    args = argparser.parse_args(arguments)
    return args


def main(arguments):
    args = parse_args(arguments)
    xml_tree = ET.parse(args.input)
    data = defaultdict(list)
    for review in xml_tree.getroot():
        rid = review.get('rid')
        for sentence in review.find('sentences'):
            sid = sentence.get('id')
            text = sentence.find('text').text
            opinions = sentence.find('Opinions')
            if opinions:
                for opinion in opinions.iterfind('Opinion'):
                    data['rid'].append(rid)
                    data['sid'].append(sid)
                    data['sentence'].append(text)
                    data['category'].append(opinion.get('category'))
                    data['target'].append(opinion.get('target'))
                    data['from'].append(opinion.get('from'))
                    data['to'].append(opinion.get('to'))
                    data['polarity'].append(opinion.get('polarity'))
    df = pd.DataFrame(data)
    file_path, _ = os.path.splitext(args.input)
    df.to_csv(file_path+'.csv', index=False)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
