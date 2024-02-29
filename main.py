import csv
import enum
import io
import json
import logging
import time
from typing import List
import warnings
import requests
from threading import Thread

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s\t%(levelname)s\t%(message)s",
	datefmt="%Y-%m-%dT%H:%M:%S",
)

FINAM_URL = "http://export.finam.ru/"

class Interval(enum.IntEnum):
	TICK = 1
	MIN_1 = 2
	MIN_5 = 3
	MIN_10 = 4
	MIN_15 = 5
	MIN_30 = 6
	HOUR = 7
	DAY = 8
	WEEK = 9
	MONTH = 10

class PriceAt(enum.IntEnum):
	OPEN = 0
	CLOSE = 1

with open("./tickers.json", "r") as f:
	tickers = json.load(f)

def _convert(row):
	if "date" not in row:
		print(row)
	date = dt.datetime.strptime(row["date"], "%Y%m%d")
	time = dt.datetime.strptime(row["time"], "%H%M%S").time()
	return {
		"datetime": dt.datetime.combine(date, time),
		"open": float(row["open"]),
		"close": float(row["close"]),
		"high": float(row["high"]),
		"low": float(row["low"]),
		"vol": float(row["vol"]),
	}

def _fetch_one(
	ticker: str,
	start_date: dt.date,
	end_date: dt.date,
	interval: Interval = Interval.DAY,
	price_at: PriceAt = PriceAt.OPEN
) -> List[dict]:
	date_format_id = 1
	time_format_id = 1

	is_moscow_tz = True
	correct_moscow_time = True
	header_required = False
	comma_fields_sep_id = 1
	no_radix_sep_format_id = 1
	row_format_id = 1
	row_fields = ["ticker", "per", "date", "time", "open", "high", "low", "close", "vol"]

	def get_response():
		return requests.get(
			f"{FINAM_URL}/{ticker}",
			params={
				"market": 0,
				"em": tickers[ticker],
				"code": ticker,
				"from": start_date,
				"yf": start_date.year,
				"mf": start_date.month-1,
				"df": start_date.day,
				"to": end_date,
				"yt": end_date.year,
				"mt": end_date.month-1,
				"dt": end_date.day,
				"p": interval.value,
				"cn": ticker,
				"dtf": date_format_id,
				"tmf": time_format_id,
				"MSOR": price_at.value,
				"mstime": "on" if is_moscow_tz else "off",
				"mstimever": int(correct_moscow_time),
				"sep": comma_fields_sep_id,
				"sep2": no_radix_sep_format_id,
				"datf": row_format_id,
				"at": int(header_required),
			},
		)
	resp = get_response()
	resp.raise_for_status()
	ATTEMPTS = 3
	DELAY = dt.timedelta(seconds=5).seconds
	for i in range(ATTEMPTS):
		try:
			data = resp.content.decode("utf-8").strip()
			if not data:
				logging.warning(f"Got empty response for ticker {ticker}. Retrying request")
				time.sleep(DELAY)
				continue
			break
		except UnicodeDecodeError as e:
			body = resp.content.decode("cp1251")
			logging.warning(f"Got exception for ticker {ticker} - can't parse json from body\n:{body}")
			time.sleep(DELAY)
		except Exception as e:
			logging.warning(f"Got exception while fetching data for ticker {ticker}. Retrying request")
			time.sleep(DELAY)
	else:
		raise Exception(f"All attempts to fetch data for ticker {ticker} are exhausted")

	reader = csv.reader(io.StringIO(data), delimiter=",")
	return [_convert(dict(zip(row_fields, line))) for line in reader]


def get_date_chunks(start_date: dt.date, end_date: dt.date, delta: dt.timedelta):
	result = []
	left = start_date
	while left < end_date:
		right = left + delta
		right = min(right, end_date)
		result.append((left, right))
		left = right + dt.timedelta(days=1)
	return result


def get_file_name(ticker, name):
	return f"data-{ticker.lower()}-{name}.jsonl"

def fetch_data(tickers: list[str]):
	intervals = [
		(Interval.MIN_1, "minutely", dt.timedelta(weeks=4)),
		(Interval.DAY, "daily", dt.timedelta(weeks=4)),
	]
	for interval, name, delta in intervals:
		chunks = get_date_chunks(
			dt.date(2014, 1, 1),
			dt.date(2024, 1, 1),
			delta
		)
		for t in tickers:
			file_name = get_file_name(t, name)
			with open(file_name, "w+") as f:
				for e in chunks:
					logging.info(f"Fetching ticker {t}, interval {name}, chunk {e}")
					rows = _fetch_one(t, *e, interval)
					for row in rows:
						json.dump(row, f, ensure_ascii=False, default=str)
						f.write("\n")
			logging.info(f"Done fetching ticker {t} for interval {name}, file {file_name} is ready")


def get_features(x, from_file):
	import tsfresh
	from tsfresh.utilities.dataframe_functions import impute
	if from_file:
		extracted_features = pd.read_pickle("features.pickle", compression="zstd")
		logging.info("Loaded tsfresh features from file")
	else:
		extracted_features = tsfresh.extract_features(
			x,
			# column_id='ticker',
			# column_sort='datetime',
			column_id='id',
			column_sort='id',
		)
		extracted_features = impute(extracted_features)
		logging.info("Dumping features to file")
		extracted_features.to_pickle("features.pickle", compression="zstd")

	logging.info(f"Extracted features:\n{extracted_features}\n{extracted_features.dtypes}")
	return extracted_features

def get_arrays(df: pd.DataFrame):
	target_field = 'close'
	x = df.drop([target_field], axis="columns")
	y = df[target_field]
	extracted_features = get_features(x, from_file=False)

	return (extracted_features, y)


def train(df: pd.DataFrame):
	from sklearn.svm import SVR
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LinearRegression
	import matplotlib.pyplot as plt

	x, y = get_arrays(df)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/6, shuffle=False)

	y_test = y_test.reset_index(drop=True)
	logging.info(f"y real:\n{y_test}")

	# ??? The dual coefficients or intercepts are not finite. The input data may contain large values and need to be preprocessed.
	# model = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
	model = LinearRegression()
	logging.info("Executing model.fit ")
	model.fit(x_train, y_train)
	y_predicted = model.predict(x_test)
	logging.info(f"y predicted:\n{y_predicted}")

	plt.plot(y_predicted, label = "y predicted")
	plt.plot(y_test.values, label = "y real")
	plt.legend()
	plt.show()


def explore_target(ticker: str):
	df = pd.read_json(get_file_name(ticker, "daily"), lines=True)

	df = df.sort_values(["datetime"], ascending=[True])
	df = df.drop_duplicates(subset=['datetime'])
	df.set_index('datetime', inplace=True)
	df = df.asfreq('d')
	df = df.interpolate()

	window_sizes = [dt.timedelta(days=i) for i in range(1, 57)]
	for w in window_sizes:
		days = w.days
		indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=days)
		df[f'max_over_{days}'] = df['high'].rolling(window=indexer).max()
		df[f'min_over_{days}'] = df['low'].rolling(window=indexer).min()
		for p in ('max', 'min'):
			# TODO: open?
			df[f'{p}_pct_change_over_{days}'] = (df[f'{p}_over_{days}'] - df['open']) / df['open']


	names = [f'{p}_pct_change_over_{w.days}' for w in window_sizes for p in ('max', 'min')]

	TRADING_DAYS = dt.timedelta(weeks=12).days
	scaled_mean_column = f'mean_scaled_to_{TRADING_DAYS}'
	desc = df[names].describe()
	desc = desc.transpose()
	desc['days'] = desc.index.str.extract('(\d+)', expand=False).astype(int)
	desc[scaled_mean_column] = desc['mean'] * TRADING_DAYS / desc['days']

	group_by_days = desc.groupby("days", group_keys=True)[[scaled_mean_column]].sum()
	return (group_by_days.index, group_by_days[scaled_mean_column])


def main():
	warnings.filterwarnings("ignore")
	tickers = [
		"SBER",
		"GAZP",
		"PIKK",
		"VTBR",
		"MVID"
	]
	# fetch_data(tickers)
	plt.figure()
	for ticker in tickers:
		x, y = explore_target(ticker)
		plt.plot(x, y, label=ticker)
	plt.xlabel("Window size, days")
	plt.ylabel("Sum of max and min scaled means, pct")
	plt.legend()
	plt.grid()
	plt.show()

	# print(df.head(50).drop(['close', 'low', 'vol'], axis=1))



	# df = df.reset_index(drop=True)
	# # ???
	# # numpy.core._exceptions._UFuncBinaryResolutionError: ufunc 'add' cannot use operands with types dtype('<M8[ns]') and dtype('<M8[ns]')
	# df = df.drop(["datetime"], axis="columns")
	# df["id"] = df.index + 1
	# df["ticker_id"] = df["ticker"].apply(lambda x: tickers[x])
	# df = df.drop(["ticker"], axis="columns")
	# logging.info(f"Dataset:\n{df}\n{df.dtypes}")
	# train(df)

if __name__ == "__main__":
	main()