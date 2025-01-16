from datetime import datetime, timedelta
test_time = datetime.now()
# test_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(test_time)
test_time = test_time.replace(year=2024, month=11, day=1, hour=0, minute=0, second=0)

print(test_time.strftime("%Y-%m-%d %H:%M:%S"))

