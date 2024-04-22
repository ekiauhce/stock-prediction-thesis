import csv
import enum
import io
import json
import logging
import time
from typing import List
import warnings
import requests

import datetime as dt

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


def main():
	warnings.filterwarnings("ignore")
	tickers = [
		"SBER",
		"GAZP",
		"PIKK",
		"MVID",
		"TATN",
		"MTSS"
	]

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

if __name__ == "__main__":
	main()