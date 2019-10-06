import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os


def plot(result_dir='ray_results/gym'):
	paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(result_dir)) for f in fn if 'csv' in f]
	rewards = []
	for path in paths:
		df = pd.read_csv(path, sep=',', engine='python', header=0)
		plt.plot(df['training/episode-reward-mean'], label=path.split('/')[4])
	plt.legend(loc='lower right')
	plt.xlabel('steps')
	plt.ylabel('average return')
	plt.savefig('rewards.png')


if __name__ == '__main__':
	plot()