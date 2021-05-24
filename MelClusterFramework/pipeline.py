#7:37

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import re

#analyzes data and assigns windows as necessary
#assumes the first flag (flag0) is if a beat is started
def win_func(pipeline, channels, feature_space):
	
	#helper function for adding idx to list
	def add_window(ls, idx):
		if ls is None:
			return [idx]
		elif idx in ls:
			return sorted(ls)
		else:
			ls.append(idx)
			return sorted(ls)

	#assigning window indexes
	for i in range(len(channels)):

		channel = channels[i]

		#assigning new beat beginnings to a window
		new_beats = []
		for rowIdx,row in channel[channel['windows'].isnull()][channel['flag0']].iterrows():
			new_beats.append(rowIdx)
			channel.loc[rowIdx]['windows'] = add_window(row['windows'], win_func.win_idx)
			win_func.win_idx = win_func.win_idx+1

		#checking beat beginnings for previous beats, one window away. Adding their window ID
		for idx in channel.index:

			scope = channel.loc[:idx].tail(pipeline.window_size)

			#adding all windows in scope to each datum
			for windows in scope[scope['flag0']].windows:
				channel.loc[idx]['windows'] = add_window(channel.loc[idx]['windows'], max(windows))

		channels[i] = channel

	return channels, feature_space

win_func.win_idx=0

#amplitude and centroid featurization and scoring
#scores using the average Manhattan distance between a point and all other points
def score_func(pipeline, channels, feature_space):

	window_size = pipeline.window_size
	num_feat = window_size * 2
	input_size = pipeline.input_size

	#helper functions that performs featurization
	#accepts a dataframe, and returns the centroid and magnitude at each time step
	def featurize(window):

		#trimming to only input columns
		window = window[window.columns[:input_size]]
		
		#calculating magnitude
		magnitude = window.mean(axis=1)

		#calculating centroid
		denom = window.sum(axis=1)
		for i,_ in enumerate(window.columns):
			window.iloc[:,i]  *= i
		numerator = window.sum(axis=1)
		centroid = (numerator/denom)/input_size
	
		#aggregating output
		return np.hstack([magnitude.values,centroid.values,[None]])

	#creating a new dataframe for first run
	if feature_space is None:
		columns = ['feat{}'.format(i) for i in range(num_feat)] \
		+ ['score']

		feature_space = pd.DataFrame(columns = columns)

	#finding windows that are not already in the feature space, and adding them
	for i in range(len(channels)):
		channel = channels[i]

		#getting full windows that don't exist in the feature space
		is_new_id = []
		for idx, windows in channel['windows'].iteritems():

			if windows is None:
				continue

			#is a new, complete window
			if len(channel)-idx > window_size:

				#list of windows contains a new window
				if any([window not in feature_space.index.tolist() for window in windows]):
					for window in windows:
						if window not in is_new_id: is_new_id.append(window)

		#featurizing each new window
		for new_id in is_new_id:
			mask = channel.windows.apply(lambda x: x is not None and new_id in x)
			window = channel[mask]

			featurized_window = featurize(window)
			feature_space.loc[new_id] = featurized_window

	#scoring all unscored points in the feature space
	noscore = feature_space.drop(columns = ['score'])
	for i, row in feature_space.iterrows():
		if row.score is None:
			point = noscore.loc[i]
			dist = noscore.sub(point)
			score = dist.sum(axis=1).mean()
			feature_space['score'][i] = abs(score)

	return channels, feature_space

def syn_func(pipeline, channels, feature_space):
	return channels, feature_space

#gets all the windows in all provided channels
#expected to be a list
def get_windows(channels):

	all_windows = []

	for channel in channels:
		windows = channel.windows
		for window_group in windows:
			for winid in window_group:
				if winid not in all_windows:
					all_windows.append(winid)

	return all_windows

#creates a TSNE plot of the feature_space
#accepts arguments seperated by spaces:
#"p s[1:4]" would plot the mel spectrogram, and produce a score for channels 1-3
def async_func(pipeline, channels, feature_space, args):

	if any(['t' in arg for arg in args]):
		print('plotting with p switch')
		
		#calculating tsne
		features = feature_space.drop('score',axis = 1)
		features_embeded = TSNE(n_components=2).fit_transform(features.values)

		#creating plotting df
		plot_df = pd.DataFrame(features_embeded, columns=['x', 'y'])

		#setting window ID and scores
		plot_df['window_id']=feature_space.index
		plot_df['score']=feature_space.score

		#finding origin channel of each window id
		win_chan = {}
		for channelid, channel in enumerate(channels):
			winids = []
			for sublist in channel.windows.values:
				if sublist is None:
					continue
				for item in sublist:
					if item not in winids: 
						winids.append(item)

			for winid in winids:
				win_chan[winid] = channelid
		plot_df['channel_id'] = [win_chan[winid] for winid in plot_df['window_id']]

		sns.scatterplot(data=plot_df, x="x", y="y", size='score', hue='channel_id')
		plt.show()

	#getting the score of a slice [1:3], single channel [2], or all []
	for arg in args:
		if 's' in arg:

			#parsing arguments
			result = re.search('s(.*)', arg).group(1)
			result = result.replace('[', '')
			result = result.replace(']', '')

			#getting windows to find the scores of
			if ':' in result:
				i1 = int(result.split(':')[0])
				i2 = int(result.split(':')[1])
				windows = get_windows(channels[i1:i2])
			elif len(result) == 0:
				windows = get_windows(channels)
			else:
				windows = get_windows([channels[int(result)]])

			#eliminating windows without scores
			windows = [w for w in windows if w in feature_space.index]

			#printing the score
			sys.stdout.write(str(feature_space.loc[windows].score.mean()))


	
	return channels, feature_space

#a pipeline for excepting time series 2D data in channels, grouping data into "beats",
#and processing and scoring steps
class Pipeline:
	def __init__(self):

		#parameters
		self.num_channels = None
		self.input_size = None
		self.channel_length = None
		self.num_flags = None
		self.window_size = None
		self.win_function = None
		self.score_function = None
		self.syncronous_function = None
		self.asyncronous_function = None

		self.flags = None

		#internal structures
		self.channels = [] #list of dfs
		self.feature_space = None #df

	#accept commands in the form of strings
	def cmd(self,cmd):

		if 'initialize' in cmd:
			self.initialize()

		#setting variables
		if 'set' in cmd:
			variable = cmd.split(' ')[1]
			value = cmd.split(' ')[2:][0]

			snip = 'self.{}={}'.format(variable, value)

			exec(snip)

			if 'set flags' in cmd and num_flags is None:
				num_flags = len(flags)

		if 'add' in cmd:
			indata = ''.join(cmd.split()[1:])
			data = eval(indata)
			self.add_data(data[0], data[1], data[2])

		if 'run asynch' in cmd:

			self.run_async(cmd.split()[2:])

		if 'echo' in cmd:
			sys.stdout.write(cmd)

	#set up the data structures based on the parameters
	def initialize(self):

		#creating the dataframes which holds channels
		columns = ['input{}'.format(i) for i in range(self.input_size)] \
		+ ['flag{}'.format(i) for i in range(self.num_flags)]\
		+ ['windows']

		self.channels = [None]*self.num_channels
		for idx in range(self.num_channels):
			self.channels[idx] = pd.DataFrame(columns = columns)

	def add_data(self, channel, data, flags):

		#adding data
		if len(data) != self.input_size:
			raise ValueError('received {} data points in a channel with length {}'.format(len(data), self.input_size))

		if len(flags) != self.num_flags:
			raise ValueError('received {} data flags, expected {}'.format(len(flags), self.num_flags))

		app_dict = {}
		for i in range(len(data)):
			app_dict['input{}'.format(i)] = data[i]

		for i in range(len(flags)):
			app_dict['flag{}'.format(i)] = flags[i]

		app_dict['windows'] = None

		self.channels[channel] = self.channels[channel].append(app_dict, ignore_index=True)
		self.channels[channel] = self.channels[channel].tail(self.channel_length)

		#running window creation
		self.run_win()

		#scoring windows
		self.run_score()

		#running syncronous function
		self.run_sync()

	def run_win(self):
		self.channels, self.feature_space = self.win_function(self, self.channels, self.feature_space)

	def run_score(self):
		self.channels, self.feature_space = self.score_function(self, self.channels, self.feature_space)

	#runs the synchronous function
	def run_sync(self):
		self.channels, self.feature_space = self.syncronous_function(self, self.channels, self.feature_space)

	#runs the synchronous function
	def run_async(self, args):
		self.channels, self.feature_space = self.asyncronous_function(self, self.channels, self.feature_space, args)

def test():

	p = Pipeline()
	p.cmd('set num_channels 2')
	p.cmd('set num_flags 2')
	p.cmd('set input_size 5')
	p.cmd('set channel_length 100')
	p.cmd('set syncronous_function syn_func')
	p.cmd('set asyncronous_function async_func')
	p.cmd('set win_function win_func')
	p.cmd('set window_size 5')
	p.cmd('set score_function score_func')
	p.cmd('initialize')

	p.cmd('add [0, [0,1,2,3,4], [False, 1]]')

	p.add_data(0, [0,1,2,3,4], [False, '1'])

	p.add_data(0, [0,2,2,3,5], [True, '2'])
	p.add_data(0, [0,7,2,3,4], [False, '3'])
	p.add_data(0, [0,6,2,3,4], [False, '4'])
	p.add_data(0, [0,3,2,3,4], [False, '5'])

	p.add_data(0, [0,1,2,9,5], [True, '6'])
	p.add_data(0, [0,1,2,7,4], [False, '7'])
	p.add_data(0, [0,1,2,4,4], [False, '8'])

	p.add_data(0, [0,1,2,3,5], [True, '6'])
	p.add_data(0, [0,1,2,3,5], [True, '6'])
	p.add_data(0, [0,1,2,3,5], [True, '6'])
	p.add_data(0, [0,1,2,3,4], [False, '8'])
	p.add_data(0, [0,1,2,3,4], [False, '8'])
	p.add_data(0, [0,1,2,3,4], [False, '8'])
	p.add_data(0, [0,1,2,3,4], [False, '8'])
	p.add_data(0, [0,1,2,3,4], [False, '8'])
	p.add_data(0, [0,1,2,3,4], [False, '8'])


	print('channel 2')
	p.add_data(1, [0,1,2,3,4], [False, '1'])

	p.add_data(1, [0,1,2,3,5], [True, '2'])
	p.add_data(1, [0,1,2,3,4], [False, '3'])
	p.add_data(1, [0,1,2,3,4], [False, '4'])
	p.add_data(1, [0,1,2,3,4], [False, '5'])

	p.add_data(1, [0,1,2,3,5], [True, '6'])
	p.add_data(1, [0,1,2,3,4], [False, '7'])
	p.add_data(1, [0,1,2,3,4], [False, '8'])

	p.add_data(1, [0,1,2,3,5], [True, '6'])
	p.add_data(1, [0,1,2,3,5], [True, '6'])
	p.add_data(1, [0,1,2,3,4], [False, '8'])

	p.cmd('run asynch')

	print(p.channels)
	print(p.feature_space)

def test2():
	p = Pipeline()
	p.cmd('set num_channels 33')
	p.cmd('set num_flags 2')
	p.cmd('set input_size 6')
	p.cmd('set channel_length 100')
	p.cmd('set syncronous_function syn_func')
	p.cmd('set asyncronous_function async_func')
	p.cmd('set win_function win_func')
	p.cmd('set window_size 5')
	p.cmd('set score_function score_func')
	p.cmd('initialize')

	p.cmd('add 0 ,[1, 2, 1, 2, 1, 5] ,[True, 1]')
	p.cmd('add 0 ,[1, 2, 1, 2, 1, 5] ,[False, 1]')
	p.cmd('add 0 ,[1, 2, 1, 2, 1, 5] ,[False, 1]')
	p.cmd('add 0 ,[1, 2, 1, 2, 1, 5] ,[False, 1]')
	p.cmd('add 0 ,[1, 2, 1, 2, 1, 5] ,[True, 1]')
	p.cmd('add 0 ,[1, 2, 1, 2, 1, 5] ,[False, 1]')
	p.cmd('add 0 ,[1, 3, 1, 2, 1, 5] ,[True, 1]')
	p.cmd('add 0 ,[1, 2, 1, 2, 1, 5] ,[False, 1]')
	p.cmd('add 0 ,[1, 3, 1, 2, 1, 5] ,[False, 1]')
	p.cmd('add 0 ,[1, 2, 1, 2, 1, 5] ,[False, 1]')
	p.cmd('add 0 ,[1, 2, 1, 2, 1, 5] ,[False, 1]')

	print(p.channels)
	print(p.feature_space)

	p.cmd('run asynch t s[0:4]')

def run():
	p = Pipeline()
	print('running...')

	while True:
		print('input: ')
		cmd = sys.stdin.readline()
		sys.stdout.flush()
		# cmd = input()
		p.cmd(cmd)

if __name__ == '__main__':
	run()