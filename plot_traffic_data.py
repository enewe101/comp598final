from matplotlib.patches import Ellipse
import copy
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np

try:
	from sklearn import cross_validation
	from sklearn import svm 
except ImportError:
	pass


def read_traffic_data(filtered=True):
	fh = open('../intersection_usage/flows.json')
	data = json.loads(fh.read())
	if filtered:
		data = filter_one_direction(data)
	return data


def read_bixi_data():
	fh = open('../bixi_data/parsed/bixi_data_deMentanaLaurier_4.csv')
	reader = csv.reader(fh)
	usage = [r[4] for r in reader]
	fh.seek(0)
	time = [r[7] for r in reader]

	fig = plt.figure(figsize=(20,5))
	gs = gridspec.GridSpec(1,1)
	ax = plt.subplot(gs[0])

	ax.plot(time, usage)


def read_bixi(station_id):

	station_id = str(station_id)
	station_id = '0' * (3-len(station_id)) + station_id
	fname = '../bixi_data/station_data/station_%s.json' % station_id
	data = json.loads(open(fname).read())
	return data


def day_of_week(day_id):
	MODAY_OFFSET_FOR_DAY_0 = 1
	return (day_id + MODAY_OFFSET_FOR_DAY_0) % 7 

DAYS_OF_WEEK = [
	'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
	'saturday', 'sunday'
]
def day_name_of_week(day_id):
	return DAYS_OF_WEEK[day_of_week(day_id)]


def plot_bixi_usage(
		station,
		day_idxs=None,
		start_idx=None,
		end_idx=None,
		show='both'	# 'weekends' | 'weekdays' | 'both'
	):
	data = read_bixi(station)

	# day_idx can be a single day index (integer) or a list thereof
	if isinstance(day_idxs, int):
		day_idxs = [day_idxs]

	if day_idxs is not None:
		usages = [data['data'][d][start_idx:end_idx] for d in day_idxs]

	else:
		usages = [d[start_idx:end_idx] for d in data['data']]


	# create a new plot
	fig = plt.figure(figsize=(20,5))
	gs = gridspec.GridSpec(1,1)
	ax = plt.subplot(gs[0])
	ax.set_title('station %d' % station)

	# plot data
	for i, usage in enumerate(usages):

		if day_idxs is not None:
			ax.plot(range(len(usage)), usage)

		else:
			if day_of_week(i) < 5 and (show in ['weekdays', 'both']):
				ax.plot(range(len(usage)), usage, color='b', alpha=0.1)
			elif show in ['weekends', 'both']:
				ax.plot(range(len(usage)), usage, color='r', alpha=0.25)
		
	



def show_reloading():
	plot_bixi_usage(165,[80,81,82],1000,1100)
	plot_bixi_usage(165,[80,81,82],600,650)


def show_artifacts():
	plot_bixi_usage(165,85)


def show_missed_rebalancing():
	data = p.plot_bixi_usage(120,91)


def make_rebalance_histogram(stations, smooth_factor=1):

	fill_event_holder = []
	empty_event_holder = []

	rebalance_events = read_rebalance_events()

	for i in stations:

		fill_events = np.zeros(24 * 60)
		empty_events = np.zeros(24 * 60)

		try:
			days = rebalance_events[str(i)]
		except KeyError:
			print 'warning, station %d does not exist.' % i
			continue

		for day in days:
			day_id, date, events = day

			for e in events:
				start_time, end_time, delta = e
				if delta > 0:
					fill_events[start_time] += 1
				else:
					empty_events[start_time] += 1

		fill_events = smooth(fill_events, smooth_factor)
		empty_events = smooth(empty_events, smooth_factor)
		fill_event_holder.append(fill_events)
		empty_event_holder.append(empty_events)

	return fill_event_holder, empty_event_holder


def make_rebalance_histogram_pooled(stations=None):

	fill_events = np.zeros(24 * 60)
	empty_events = np.zeros(24 * 60)
	rebalance_events = read_rebalance_events()

	jobsize = float(len(empty_events))

	i = 0
	for i, key in enumerate(rebalance_events):

		#if i % 10 == 0:
		#	print '%2.1f %%' % (100 * i / jobsize)

		station_id = int(key)
		if stations is not None:
			if station_id not in stations:
				continue

		for day in rebalance_events[key]:
			day_id, date, events = day

			for e in events:
				start_time, end_time, delta = e
				if delta > 0:
					fill_events[start_time] += 1
				else:
					empty_events[start_time] += 1

	return fill_events, empty_events


TIME_0 = 400
def plot_rebalance_events(smooth_factor=1, stations=None):

	if stations == 'each':
		stations = range(332)


	if stations is not None:
		fill_event_holder, empty_event_holder = (
			make_rebalance_histogram( stations, smooth_factor)
		)

	else:
		fill_events, empty_events = make_rebalance_histogram_pooled()
		fill_event_holder = [smooth(fill_events, smooth_factor)]
		empty_event_holder = [smooth(empty_events, smooth_factor)]

	fig = plt.figure(figsize=(10,10))
	gs = gridspec.GridSpec(1,1)
	ax = plt.subplot(gs[0])

	X = range(len(fill_event_holder[0]))

	# rotate the time axis to put 6:15 at time zero
	X = X[-TIME_0:] + X[:-TIME_0]

	jobsize = float(len(fill_event_holder))

	i = 0
	alpha = 1/float(len(fill_event_holder))
	for fills, empties in zip(fill_event_holder, empty_event_holder):

		# show_progress
		i += 1
		if i % 10 == 0:
			print '%2.1f %%' % (100 * i / jobsize)

		empties = -1 * empties
		ax.bar(X, fills, color='blue', alpha=alpha)
		ax.bar(X, empties, color='red', alpha=alpha)

	if stations is None:
		ax.set_title('All stations')
	else: 
		ax.set_title('station %d' % stations[0])

	plt.show()
	

def smooth(series, smooth_factor):
	if not isinstance(smooth_factor, int):
		raise ValueError('the smoothing factor must be an integer.')

	size = len(series)
	smoothed_size = int(np.ceil(size / float(smooth_factor)))
	smoothed = np.zeros(smoothed_size)

	for i, j in enumerate(range(0, len(series), smooth_factor)):
		smoothed[i] = sum(series[j:j + smooth_factor])

	return smoothed


def characterize_station_rebalancing():

	station_rebalance_characteristics = []
	OUTFNAME = '../bixi_data/station_rebalance_signature.csv'
	writer = csv.writer(open(OUTFNAME,'w'))

	rebalances = read_rebalance_events()
	jobsize = float(len(rebalances))
	for i, key in enumerate(rebalances):

		# print progress
		if i % 20 == 0:
			print '%2.1f %%' % (i * 100 / jobsize)

		station_id = int(key)
		days = rebalances[key]

		num_days = float(len(days))

		fills, empties = [],[]

		# get out all of the events, and sort them into fills and empties
		for day in days:
			day_id, date, events = day
			fills.extend([(e[0], e[2]) for e in events if e[2]>0])
			empties.extend([(e[0], e[2]) for e in events if e[2]<0])

		avg_fill_time = np.mean([f[0] for f in fills])
		std_fill_time = np.std([f[0] for f in fills])
		total_fill_vol = sum([f[1] for f in fills]) / num_days

		avg_empty_time = np.mean([f[0] for f in empties])
		std_empty_time = np.std([f[0] for f in empties])
		total_empty_vol = - sum([f[1] for f in empties]) / num_days
		

		fill_extent = np.sqrt(total_empty_vol ** 2 + total_fill_vol ** 2)
		fill_tendency = np.arctan(total_fill_vol/total_empty_vol) / (np.pi/2.)

		station_rebalance_characteristics.append((
			station_id, avg_fill_time, std_fill_time, total_fill_vol,
			avg_empty_time, std_empty_time, total_empty_vol, fill_tendency,
			fill_extent
		))

	station_rebalance_characteristics.sort()
	for src in station_rebalance_characteristics:
		writer.writerow(src)


def cross_val_rebalance(sigma=0.02, kernel='rbf', C=1, gamma=1e-4, CV=5):

	fname = '../bixi_data/station_rebalance_signature.csv'
	inputs = []
	outputs = []

	station_chars = read_station_features(sigma)

	for row in csv.reader(open(fname)):
		station_id = row[0]
		inputs.append((
			station_chars[int(station_id)]['tide'],
			station_chars[int(station_id)]['bulk'],
			station_chars[int(station_id)]['focus'],
			station_chars[int(station_id)]['churn'],
		))

		# fill_ratio
		outputs.append(float(row[7])>0.5)

	# do cross-validation of an svm classifier
	scores = []
	clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
	scores.extend(cross_validation.cross_val_score(
    	clf, inputs, outputs, cv=CV))





def plot_rebalance_characteristics(dim1, dim2, sigma=0.02):

	fname = '../bixi_data/station_rebalance_signature.csv'
	stations = []

	station_chars = read_station_features(sigma)

	for row in csv.reader(open(fname)):
		station_id = row[0]
		stations.append({
			'id': station_id,
			'lat': station_chars[int(station_id)]['lat'],
			'lon': station_chars[int(station_id)]['lon'],
			'tide': station_chars[int(station_id)]['tide'],
			'bulk': station_chars[int(station_id)]['bulk'],
			'focus': station_chars[int(station_id)]['focus'],
			'churn': station_chars[int(station_id)]['churn'],
			'avg_fill_time': float(row[1]),
			'std_fill_time': float(row[2]),
			'fill_vol': float(row[3]),
			'avg_empty_time': float(row[4]),
			'std_empty_time': float(row[5]),
			'empty_vol': float(row[6]),
			'fill_ratio': float(row[7]) > 0.5,
			'fill_extent': float(row[8])
		})

	X = [r[dim1] for r in stations]
	Y = [r[dim2] for r in stations]

	fig = plt.figure(figsize=(10,10))
	gs = gridspec.GridSpec(1,1)
	ax = plt.subplot(gs[0])

	ax.plot(X,Y,linestyle='None', marker='.')



def show_rebalance_types():
	plot_rebalance_characteristics('fill_vol', 'empty_vol')

def show_empty_times():
	plot_rebalance_characteristics('avg_empty_time','empty_vol')

def show_fill_times():
	plot_rebalance_characteristics('avg_fill_time', 'fill_vol')


def read_rebalance_events():
	FNAME = '../bixi_data/rebalance_events.json'
	data = json.loads(open(FNAME).read())
	return data


def extract_rebalance_events():
	INPUT_DIR = '../bixi_data/station_data'
	OUTPUT_DIR = '../bixi_data/rebalance_events.json'

	out_fh = open(OUTPUT_DIR, 'w')

	fnames = os.popen('ls %s' % INPUT_DIR).read().split()
	jobsize = float(len(fnames))

	rebalance_detector = RebalanceDetector()
	output_data = {}

	for i, fname in enumerate(fnames):
		if i % 10 == 0:
			print '%2.1f %%' % (100 * i / jobsize)


		station_id = int(fname[8:11])
		this_station = []
		output_data[station_id] = this_station

		path = os.path.join(INPUT_DIR, fname)
		data = json.loads(open(path).read())
		days = data['days']
		usage = data['data']
		for j in range(len(days)):
			rebalance_events = rebalance_detector.scan(usage[j])
			if len(rebalance_events) > 0:
				this_station.append((j, days[j], rebalance_events))

	out_fh.write(json.dumps(output_data, indent=2))


class RebalanceDetector(object):
	SIGNIFICANCE_THRESHOLD = 8

	def __init__(self):
		self.reset()


	def reset(self):
		self.events = []
		self.start_point = None
		self.state = None

	def filter_artifacts(self, events):
		filtered = []

		i = 0
		while i < len(events):

			is_close = False
			is_down_then_up = False
			is_last_event = (i == len(events) - 1)

			# if we're not at the last event, test this event and the next
			if not is_last_event:

				# unpack
				start1, stop1, delta1 = events[i]
				start2, stop2, delta2 = events[i+1]

				# do two tests
				is_close = start2 - stop1 < 10
				is_down_then_up = delta1 < 0 and delta2 > 0

			# if we arrived at the last event, it must have tested ok with the
			# next-from last.  Add it to the filtered list
			if is_last_event:
				filtered.append(events[i])
				i += 1

			# if both tests are positive, we have an artifact.
			# exclude these while advancing the index twice
			elif is_close and is_down_then_up:
				i += 2

			# the non-last event passed, so it can be added
			else:
				filtered.append(events[i])
				i += 1

		return filtered


	def scan(self, series):
		self.reset()
		for i in range(len(series) -1):

			# watch for the end of rebalancing
			if self.state == 'filling' or self.state == 'emptying':
				self.test_rebalance_continues(i, series[i], series[i+1])

			# watch for the start of rebalancing
			if self.state is None:
				self.test_rebalance_starts(i, series[i], series[i+1])

		self.events = self.filter_artifacts(self.events)

		return self.events


	def test_rebalance_starts(self, i, cur_val, next_val):

		# maybe start filling
		if next_val - cur_val > 1:
			self.state = 'filling'
			self.start_point = (i, cur_val)

		# maybe start emptying
		if next_val - cur_val < -1:
			self.state = 'emptying'
			self.start_point = (i, cur_val)


	def maybe_record_event(self, i, cur_val, next_val):
		total_change = cur_val - self.start_point[1]
		if abs(total_change) >= self.SIGNIFICANCE_THRESHOLD:
			event_start, event_end = self.start_point[0], i
			self.events.append((event_start, event_end, total_change))


	def test_rebalance_continues(self, i, cur_val, next_val):

		# if filling stopped, record it
		if self.state == 'filling':
			if not (next_val - cur_val > 1):
				self.maybe_record_event(i, cur_val, next_val)
				self.state = None

		# if emptying stopped, record it
		elif self.state == 'emptying':
			if not (next_val - cur_val < -1):
				self.maybe_record_event(i, cur_val, next_val)
				self.state = None


def filter_one_direction(intersections):

	filtered = {}
	for key in intersections:
		datum = intersections[key]
		okay = True
		for direction in ['eastbound', 'northbound', 'westbound', 'southbound']:
			for period in ['AM_flow', 'PM_flow']:
				okay = okay and (datum[period][direction] > 0)

		if okay:
			filtered[key] = copy.copy(datum)

	return filtered


def calculate_station_traffic_features(sigma, filtered=True):
	'''
		calculates the traffic features pertaining to specific bixi stations.
	'''

	FNAME = '../intersection_usage/features/focus_sigma%1.3f.csv'
	fname = FNAME % sigma
	writer = csv.writer(open(fname, 'w'))

	stations = read_bixi_stations()
	intersections = read_traffic_data(filtered=filtered)

	intersection_features = []
	jobsize = float(len(stations))
	for i, station in enumerate(stations):

		# show progress
		if i % 10 == 0:
			print '%2.1f %%' % (100 * i / jobsize)

		station_id, lat, lon = station

		tide, bulk, focus, churn = calc_station_features(
			lat,lon,intersections, sigma)

		writer.writerow((station_id, tide, bulk, focus, churn))


def calc_station_features(lat, lon, intersections, sigma=0.02):
	''' 
		helper for calculate_station_traffic_features.  Calculates the
		traffic features for one particular bixi station.
	'''

	tide = 0
	bulk = 0
	weight = 0
	focus = 0
	am_churn = []
	pm_churn = []

	i = 0
	for key in intersections:
		i += 1

		corner = intersections[key]
		corner_lat, corner_lon = corner['latitude'], corner['longitude']

		d = np.sqrt( (lat - corner_lat)**2 + (lon - corner_lon)**2 )
		w = np.e**(- 0.5 * (d / sigma)**2 )

		am_churn.append((w, corner['am_net_flow']))
		pm_churn.append((w, corner['pm_net_flow']))

		# calculate the degree to which triffic is heading toward this
		# station
		if corner['tidality'] > 0.8:
			to_station = np.array([lon - corner_lon, lat - corner_lat])
			to_station = to_station / np.linalg.norm(to_station)
			am_towards = np.dot(corner['am_net_flow'], to_station)
			pm_towards = np.dot(corner['pm_net_flow'], to_station)

			focus += w * (am_towards - pm_towards)

		tide += w * corner['tide']
		bulk += w * corner['bulk']
		weight += w

	am_avg = (
		np.mean([c[0] * c[1][0] for c in am_churn]),
		np.mean([c[0] * c[1][1] for c in am_churn])
	)

	pm_avg = (
		np.mean([c[0] * c[1][0] for c in pm_churn]),
		np.mean([c[0] * c[1][1] for c in pm_churn])
	)

	churn = np.sqrt(
		np.sum([
			c[0] * ((am_avg[0] - c[1][0])**2 + (am_avg[1] - c[1][1])**2) 
			for c in am_churn
		])
		+ np.sum([
			c[0] * ((pm_avg[0] - c[1][0])**2 + (pm_avg[1] - c[1][1])**2) 
			for c in pm_churn
		])
	)

	tide = tide / weight
	bulk = bulk / weight
	churn = churn / (2 * weight)

	return tide, bulk, focus, churn


def extract_station_ids_and_positions():
	'''
		Write a file containing just the id, latitude, and longitude
		of all bixi stations.
	'''
	DIR = '../bixi_data/station_data/'
	fnames = os.popen('ls %s' % DIR).read().split()
	OUTFILE = '../bixi_data/stations.csv'
	writer = csv.writer(open(OUTFILE, 'w'))

	stations = set()
	jobsize = float(len(fnames))
	for i, fname in enumerate(fnames):
		if i % 10 == 0:
			print '%2.1f %%' % (100 * i / jobsize)
		idx = int(fname[8:11])
		station = json.loads(open(os.path.join(DIR,fname)).read())
		lat, lon = station['lat'], station['long']
		stations.add((idx, lat, lon))

	stations = sorted(list(stations))
	for station in stations:
		writer.writerow(station)


def read_station_features(sigma=0.020):

	# read the per-station traffic features
	station_features_fh = open(
		'../intersection_usage/features/focus_sigma%1.3f.csv' % sigma)
	reader = csv.reader(station_features_fh)
	station_features = {}
	for row in reader:
		station_features[int(row[0])] = {
			'tide': float(row[1]),
			'bulk': float(row[2]),
			'focus': float(row[3]),
			'churn': float(row[4])
		}

	# read the stations list
	stations_list = read_bixi_stations()

	# merge features with stations lat / lon positions
	stations = {}
	for s in stations_list:
		station_id = s[0]
		stations[station_id] = {
			'lat': s[1], 
			'lon': s[2],
			'tide': station_features[station_id]['tide'],
			'bulk': station_features[station_id]['bulk'],
			'focus': station_features[station_id]['focus'],
			'churn': station_features[station_id]['churn']
		}

	return stations

	
def read_bixi_stations():
	FNAME = '../bixi_data/stations.csv'
	reader = csv.reader(open(FNAME))
	stations = [(int(row[0]), float(row[1]), float(row[2])) for row in reader]
	return stations


def do_plot_tides():

	MONTREAL_ANGLE = 50	# angle by which Montreal "North" deviates from North
	SCALE_FLOW = 1e-5	# multiply flow vectors to make a reasonable
								# scale

	fig = plt.figure(figsize=(10,10))
	gs = gridspec.GridSpec(1,1)
	ax = plt.subplot(gs[0])

	stations = read_bixi_stations()
	station_lats = [s[1] for s in stations]
	station_lons = [s[2] for s in stations]

	assert(False) # read_traffic_tides needs to be changed to read_traffic_data
	intersections = read_traffic_tides()
	tides = np.array([t[2] for t in intersections])
	min_tide, max_tide = min(tides), max(tides)
	tides = (tides - min_tide) / float(max_tide - min_tide)

	bulks = np.array([t[3] for t in intersections])
	min_bulk, max_bulk = min(bulks), max(bulks)
	bulks = (bulks - min_bulk) / float(max_bulk - min_bulk)

	for corner, tide, bulk in zip(intersections, tides, bulks):
		lat, lon = corner[0], corner[1]
		ax.plot(lon, lat, marker='.', linestyle='None', 
				markeredgecolor=(tide,1-tide,0), 
				markerfacecolor=(tide,1-tide,0)
		)

	ax.plot(station_lons, station_lats, marker='.', linestyle='None')

	plt.draw()


def do_plot(filtered=True, sigma=0.02, colorize='focus'):

	MONTREAL_ANGLE = 50	# angle by which Montreal "North" deviates from North
	SCALE_FLOW = 1e-5	# multiply flow vectors to make a reasonable
								# scale

	fig = plt.figure(figsize=(10,10))
	gs = gridspec.GridSpec(1,1)
	ax = plt.subplot(gs[0])

	data = read_traffic_data(filtered)

	# get out intersection locations and vecs representing their bulk am flow
	flows = []
	grey_flows = []
	for key in data:

		datum = data[key]
		am_flow = datum['am_net_flow']
		pm_flow = datum['pm_net_flow']

		# rotate and scale the net flow vectors 
		am_net_flow = [f * SCALE_FLOW for f in am_flow]
		pm_net_flow = [f * SCALE_FLOW for f in pm_flow]

		if datum['tidality'] < 0.8:
			grey_flows.append((
				datum['latitude'], datum['longitude'], 
				am_net_flow, pm_net_flow
			))
		else:
			flows.append((
				datum['latitude'], datum['longitude'], 
				am_net_flow, pm_net_flow
			))

	latitudes = [f[0] for f in flows]
	longitudes = [f[1] for f in flows]
	ax.plot(longitudes, latitudes, marker='.', linestyle='None')

	#e = Ellipse(xy=(-73.6, 45.5), width=0.02, height=0.02)
	#ax.add_artist(e)
	#e.set_facecolor('red')

	for lat, lon, am_flow, pm_flow in grey_flows:
		add_arrow( (lon,lat), am_flow, ax, color='0.75', redraw=False)
		add_arrow( (lon,lat), pm_flow, ax, color='0.55', redraw=False)

	for lat, lon, am_flow, pm_flow in flows:

		#tide = am_flow[0] + pm_flow[0], am_flow[1] + pm_flow[1]
		#flux = am_flow[0] + pm_flow[0] + am_flow[1] + pm_flow[1]

		#add_arrow( (lon,lat), tide, ax, color='green', redraw=False)
		add_arrow( (lon,lat), am_flow, ax, color='purple', redraw=False)
		add_arrow( (lon,lat), pm_flow, ax, color='magenta', redraw=False)


	# plot the bixi stations
	stations = read_station_features(sigma)
	station_lats = [s['lat'] for s in stations.values()]
	station_lons = [s['lon'] for s in stations.values()]
	station_foci = [s['focus'] for s in stations.values()]
	station_churns = [s['churn'] for s in stations.values()]

	colors = red_green_colorize([s[colorize] for s in stations.values()])

	for i in range(len(station_lats)):
		lat = station_lats[i]
		lon = station_lons[i]
		color = colors[i]
		ax.plot(
			[lon], [lat], marker='o', linestyle='None',
			markerfacecolor=color, markeredgecolor=color
		)

	plt.draw()
	return ax


def red_green_colorize(vals):

	vals = rescale(vals)

	ones = np.ones(len(vals))

	red_vals = 2*(1-vals)
	green_vals = 2*vals
	blue_vals = np.zeros(len(vals))

	reds = np.minimum(ones, red_vals)
	greens = np.minimum(ones, green_vals)
	blues = np.minimum(ones, blue_vals)

	return zip(reds, greens, blues)


def green_blue_colorize(vals):
	vals = rescale(vals)

	ones = np.ones(len(vals))

	red_vals = np.zeros(len(vals))
	green_vals = 2*(1-vals)
	blue_vals = 2*vals

	reds = np.minimum(ones, red_vals)
	greens = np.minimum(ones, green_vals)
	blues = np.minimum(ones, blue_vals)

	return zip(reds, greens, blues)



def rescale(vals):
	beta = 5
	min_val = min(vals)
	max_val = max(vals)
	vals = np.array(vals)

	if min_val == max_val:
		vals = np.zeros(len(vals))

	else:
		vals = (vals - np.mean(vals)) / float(np.std(vals))

	vals = 1 / (1 + np.e**( - beta * vals) )
	return vals

def rotate(x, y, theta_deg):

	theta_rad = np.pi * theta_deg / 180.0
	cos = np.cos(theta_rad)
	sin = np.sin(theta_rad)

	new_x = x*cos - y*sin
	new_y = x*sin + y*cos

	return new_x, new_y


def test_add_arrow():
	ax = plt.subplot(111)
	ax.plot([0.5],[0.4], linestyle='None', marker='.', markeredgecolor='red',
		markerfacecolor='red')

	plt.xlim((0, 1))
	plt.ylim((0, 1))
	add_arrow((0.5,0.4), (-0.1, -0.1), ax, redraw=True)

	plt.show()


def add_arrow(source, delta, ax, color='black', redraw=True):
	'''
		Adds an arrow at source, pointing in the direction delta, to the
		axes ax.
	'''
	# get the desired endpoint based on delta
	target = (source[0] + delta[0], source[1] + delta[1])

	# create the arrow
	arrow = ax.annotate(
		'', 
		xy=target, xytext=source, 
		arrowprops={
			'facecolor':color,
			'edgecolor': color,
			'width': 2,
			'headwidth': 3,
		}
	)

	# redraw the plot with the new arrow
	if redraw:
		plt.draw()

	return arrow

