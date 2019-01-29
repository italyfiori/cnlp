#!/usr/bin/python

import numpy as np

year = {
	'[\d+]年': '{digit}',
	'公元[\d+]年': '{digit}',
	'公元前[\d+]年': '{digit}',
}

month_of_year = {
	'[1-12]月': '{digit}',
	'的?第[1-12]个?月': '{digit}',
}

day_of_month = {
	'的?第{digit}天': '{digit}',
	'的?{digit}号': '{digit}',
}

week_of_month = {
	'第{digit}周': '{digit}',
	'{digit}周': '{digit}',
}


week_of_year = {
	'第{digit}周': '{digit}',
	'{digit}周': '{digit}',
}

day_of_week = {
	'周{digit:1-7}': '{digit}',
	'星期{digit:1-6}': '{digit}',
	'星期天': 7,
	'星期日': 7,
	'的?第{digit}天': '{digit}',
}



month_compare_today = {
	'上一个月': -1,
	'上个月': -1,
	'下一个月': 1,
	'下个月': 1,
	'本月': 0,
	'当月': 0,
}

week_compare_today = {
	'上一周': -1,
	'上周': -1,
	'上个周': -1,
	'上个星期': -1,
	'下一周': 1,
	'下周': 1,
	'下个周': 1,
	'下个星期': 1,
	'本周': 0,
}

day_compare_today = {
	'今天' : 0,
	'明天' : 1,
	'昨天' : -1,
	'后天' : 1,
	'大前天' : -2,
	'大后天' : 2,
	'大大前天' : -3,
	'大大后天' : 3,
}



hour = {
	'{digit:0-23}点': '{digit}',
	'{digit:0-23}时': '{digit}',
}

prefixs = {
	'上一个': -1,
	'下一个': 1,
	'上': -1,
	'下': 1
}

patterns = [	
	# 年*/月/日, 2018年12月12日
	# 年*/月/周/日, 2018年/12月/的第二周/的周末
	# 年*/周/日, 2018年37周第5天
	# 年*/日, 2018年/的第38天
	# diff月/日, 上个月/15日
	# diff月/周/日, 上个月/第二周/15日
	# diff周/日, 上一周/星期天, 上一周/的第二天
	# diff日, 昨天


	['year:*', 'month_of_year:*', 'day_of_month'],
	['year:*', 'week_of_year:*', 'day_of_week'],
]

