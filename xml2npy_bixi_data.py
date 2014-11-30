#Parse bixi data xml files and save to a npy file
# Uncompress all bixi_data files and delete trajectory folder prior to running this script

import xml.etree.cElementTree as ET
import xml.parsers.expat as expat

import os
import logging
import numpy 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('modellog.log')
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

PATH = '../../bixi_data/'

def process_file(path,identifier):
    logger.debug('processing file: ' + path)
    try:
        tree = ET.parse(path)
        root = tree.getroot()
	for element in root:
	    if element.tag != 'station':
		continue
	    if element.find('name').text != identifier:
		continue
	    nbBikes = int(element.find('nbBikes').text)
	    nbEmptyDocks = int(element.find('nbEmptyDocks').text)
	    total = nbBikes+nbEmptyDocks
	    if total != 0:
	        imbalance = (1.*nbBikes) / total
	        return imbalance
	return 0
    except ET.ParseError as e:
	logger.error('Error processing ' + path + ':')
	logger.exception(e)
	return None

def process_dir(path,identifier,month):
    times, imbalances = [], []
    filenames_dict = {}
    for dirname, dirnames, filenames in os.walk(path):
	if filenames != []:
	    for filename in filenames:
		if int(filename[6]) != month:
		    continue
		if int(filename[16:17]) != 0:
		    continue
		filenames_dict[filename] = dirname

    for filename in sorted(list(filenames_dict.keys()),key=numericalSort):
	fullpath = os.path.join(filenames_dict[filename],filename)
	if not fullpath.endswith('.xml'):
	    logger.debug('skipping file: ' + fullpath)
	    continue
	imbalance = process_file(fullpath,identifier)
	if imbalance != None:
	    imbalances.append(imbalance)
	    times.append(int(filename[:-7].replace("_","")))
    return times, imbalances, identifier

def numericalSort(val):
    val = val[:-7].replace("_","")
    return int(val)

def get_identifiers(fileobj):
    identifiers = []
    tree = ET.parse(fileobj)
    root = tree.getroot()
    for identifier in root.iter('name'):
	identifiers.append(identifier.text)
    return identifiers

def save_npy(path=PATH):
    identifiers = get_identifiers(path + 'BixiDataApril2012/02/2012_04_02__00_00_01.xml')
    for identifier in identifiers:
	print identifier
	for month in [4,5,6,7,8]:
	    row = process_dir(path,identifier,month)
	    print row
	    with open('bixi_data_' + identifier.replace(" ","").replace("/","") + '_' + str(month) + '.npy','wb') as f:
	        numpy.save(f,row)

def to_npy(path=PATH):
    identifiers = get_identifiers(path + 'BixiDataApril2012/02/2012_04_02__00_00_01.xml')
    ret = []
    for identifier in identifiers:
	print identifier
	for month in [4,5,6,7,8]:
	    row = process_dir(path,identifier,month)
	    ret.append(row)
    return numpy.array(ret)

if __name__=='__main__':
    save_npy(PATH)
