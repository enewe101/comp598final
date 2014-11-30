# Plotting bike count time series data for each bixi station
# Uncompress all bixi_data files and delete trajectory folder prior to running this script

import xml.etree.cElementTree as ET
import xml.parsers.expat as expat

import os
import logging

import matplotlib.pyplot as plt

DAYS_LIST = [1,6,11,16,21,26]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('plotlog.log')
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

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

def process_dir(path,identifier,month,day):
    times, imbalances = [], []
    filenames_dict = {}
    for dirname, dirnames, filenames in os.walk(path):
	if filenames != []:
	    for filename in filenames:
		if int(filename[6]) != month:
		    continue
		if int(filename[8:10]) != day:
		    continue
		if int(filename[15:17]) not in [00,30]:
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
    return times, imbalances

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

def main():
    path = '../../bixi_data/'
    identifiers = get_identifiers(path + 'BixiDataApril2012/02/2012_04_02__00_00_01.xml')
    for identifier in identifiers:
	print identifier
	for month in [4,5,6,7,8]:
	    for day in DAYS_LIST:
	        times, imbalances = process_dir(path,identifier,month,day)
	        plt.plot(imbalances,label=day)
	    plt.title('Imbalances for station ' + identifier + str(month))
	    plt.ylabel('Imbalance')
	    plt.xlabel('Time')
	    plt.legend()
	    plt.savefig(identifier.replace(' ','').replace('/','') + str(month) + '.png')
	    plt.clf()

if __name__=='__main__':
    main()
