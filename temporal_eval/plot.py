import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np


def plot_llm_accuracy(csv_path: Path, out_path: Path, show: bool = False):
	df = pd.read_csv(csv_path, sep=';', encoding='utf-8')

	# Normalize column names
	df.columns = [c.strip() for c in df.columns]

	# Expecting columns: 'LLM', 'Accuracy', 'Date', 'F1'
	if 'LLM' not in df.columns or 'Accuracy' not in df.columns or 'Date' not in df.columns:
		raise ValueError('CSV must contain columns: LLM, Accuracy, Date')

	# Parse dates (format like dd.mm.yyyy) using dayfirst
	df['Date_parsed'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
	if df['Date_parsed'].isna().any():
		# try generic parse if some failed
		df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')

	df = df.dropna(subset=['Date_parsed'])

	# Convert Accuracy to numeric
	df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
	df = df.dropna(subset=['Accuracy'])

	# Convert F1 to numeric if it exists
	if 'F1' in df.columns:
		df['F1'] = pd.to_numeric(df['F1'], errors='coerce')

	# Sort by date for nicer plotting order
	df = df.sort_values('Date_parsed')

	# Infer provider from the model name
	def infer_provider(name: str) -> str:
		s = name.lower()
		if 'claude' in s:
			return 'Anthropic'
		if 'gpt' in s:
			return 'OpenAI'
		if 'mistral' in s:
			return 'Mistral'
		if 'llama' in s or 'llama' in s:
			return 'Meta'
		if 'deepseek' in s:
			return 'DeepSeek'
		# fallback to first token
		return name.split()[0]

	df['Provider'] = df['LLM'].map(infer_provider)

	# Decide whether a model is open-source by keywords
	open_keywords = ('oss', 'mistral', 'llama', 'deepseek')
	df['IsOpen'] = df['LLM'].str.lower().apply(lambda v: any(k in v for k in open_keywords))

	providers = list(dict.fromkeys(df['Provider']))
	palette = sns.color_palette('Set2', n_colors=max(2, len(providers)))
	color_map = {p: palette[i % len(palette)] for i, p in enumerate(providers)}

	# Helper function to plot a metric without lines
	def plot_metric(ax, metric_col: str, metric_name: str):
		df_metric = df.dropna(subset=[metric_col]).copy()

		# Plot each point with provider color and marker by open/proprietary
		for _, row in df_metric.iterrows():
			x = row['Date_parsed']
			y = row[metric_col]
			prov = row['Provider']
			is_open = row['IsOpen']
			color = color_map.get(prov, 'C0')
			marker = 's' if is_open else 'o'  # square for open-source, circle for proprietary
			ax.scatter([x], [y], s=170, color=color, marker=marker)
			# center the label inside the marker
			ax.text(x, y, row['LLM'], ha='center', va='center', fontsize=10, clip_on=False)

		# Split full span into contiguous 4-month chunks starting at the earliest date
		if df_metric.empty:
			# Format x-axis for dates
			ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
			plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
			ax.set_xlabel('LLM Release Date')
			# ax.set_ylabel(metric_name)
			# ax.set_title(f'LLM {metric_name} by Release Date')
			ax.grid(axis='y', linestyle='-', alpha=0.4)
			return

		start = df_metric['Date_parsed'].min()
		# month index from year 0
		end = df_metric['Date_parsed'].max()
		end_idx = end.year * 12 + end.month
		month_idx = df_metric['Date_parsed'].dt.year * 12 + df_metric['Date_parsed'].dt.month

		# subtract from the end and floor divide
		df_metric['Chunk4M'] = ((end_idx - month_idx) // 6).astype(int)

		# For open-source and proprietary, select best model within each 4-month chunk and connect them
		for is_open, label, color, linestyle in [
			(True, 'Open-source', '#2ecc71', '--'),
			(False, 'Proprietary', '#e74c3c', '--'),
		]:
			sub = df_metric[df_metric['IsOpen'] == is_open]
			if sub.empty:
				continue
			# choose index of max metric in each chunk
			idx = sub.groupby('Chunk4M')[metric_col].idxmax().dropna()
			selected = sub.loc[idx].sort_values('Date_parsed')
			# drop duplicate metric values within same chunk (keep first) — optional
			selected = selected.drop_duplicates(subset=['Chunk4M'], keep='first')
			if not selected.empty:
				ax.plot(selected['Date_parsed'].values, selected[metric_col].values,
						color=color, linestyle=linestyle,
						linewidth=1, alpha=0.9, label=f'Best {label.lower()}', zorder=3)
				# for _, r in selected.iterrows():
					# ax.annotate(r['LLM'], (r['Date_parsed'], r[metric_col]),
								# xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)

		# Format x-axis for dates
		ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
		plt.setp(ax.get_xticklabels(), ha='center')

		ax.set_xlabel('LLM Release Date')
		# ax.set_ylabel(metric_name)
		ax.set_ylim(0.48, 0.71)
		ax.set_title(f'{metric_name}', fontweight='bold')
		ax.grid(axis='y', linestyle='-', alpha=0.4)
		# remove per-axis legend to avoid duplicate/line entries

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
 	

	# Plot Accuracy on the left
	plot_metric(ax1, 'Accuracy', 'Accuracy')

	# Plot F1 on the right if it exists
	if 'F1' in df.columns and df['F1'].notna().any():
		plot_metric(ax2, 'F1', 'F1')
	else:
		ax2.text(0.5, 0.5, 'F1 data not available', ha='center', va='center', transform=ax2.transAxes)
		ax2.set_title('F1 Data')
  
	fig.supylabel('Metric')

	# Create combined legend with titled sections for Provider (colors) and Type (markers)
	title_provider = plt.Line2D([0], [0], marker=None, linestyle='None', color='none', label='Provider')
	provider_handles = [
		plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[p], markeredgecolor='k', markersize=8, label=p)
		for p in providers
	]
	title_type = plt.Line2D([0], [0], marker=None, linestyle='None', color='none', label='Type')
	marker_handles = [
		plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', markeredgecolor='k', markersize=8, label='Open-source'),
		plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markeredgecolor='k', markersize=8, label='Proprietary'),
	]

	# add explicit line handles for the 6-month best-model lines
	line_handles = [
		plt.Line2D([0], [0], color='#2ecc71', linestyle='--', linewidth=1, label='Best open-source'),
		plt.Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=1, label='Best proprietary'),
	]
	# title for trend section
	title_trend = plt.Line2D([0], [0], marker=None, linestyle='None', color='none', label='Trend')

	all_handles = [title_provider] + provider_handles + [title_type] + marker_handles + [title_trend] + line_handles
	all_labels = ['Provider'] + providers + ['Type', 'Open-source', 'Proprietary', 'Trend', 'Best open-source', 'Best proprietary']
	leg = fig.legend(handles=all_handles, labels=all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4)
	# Make the section titles bold
	texts = leg.get_texts()
	if texts:
		texts[0].set_fontweight('bold')  # "Provider"
		type_idx = 1 + len(providers)
		if type_idx < len(texts):
			texts[type_idx].set_fontweight('bold')  # "Type"
		# Bold the "Trend" title if present
		try:
			trend_idx = all_labels.index('Trend')
		except ValueError:
			trend_idx = None
		if trend_idx is not None and trend_idx < len(texts):
			texts[trend_idx].set_fontweight('bold')

	fig.tight_layout()
	out_dir = out_path.parent
	out_dir.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description='Plot LLM accuracy over release date')
	parser.add_argument('--csv', type=Path, default=Path('llm_results_1.csv'))
	parser.add_argument('--out', type=Path, default=Path('llm_by_release_6.svg'))
	parser.add_argument('--show', action='store_true')
	args = parser.parse_args()

	if not args.csv.exists():
		raise FileNotFoundError(f'CSV file not found: {args.csv}')

	plot_llm_accuracy(args.csv, args.out, show=args.show)


if __name__ == '__main__':
	main()

