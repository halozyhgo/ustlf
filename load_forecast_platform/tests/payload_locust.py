from locust import HttpUser, task, between


class VppUser(HttpUser):
    host = "http://127.0.0.1:5000"
    wait_time = between(10000,10001)

    def on_start(self):
        print("start to test")

    def on_stop(self):
        print("stopping...")

    @task
    def payload(self):
        url = "/ustlf/station/real_time_data_upload"
        data = {
    "site_id": "123",

    "real_his_load":[
        {
            "load_time": "2024-09-02 04:00:00",
            "load_data": 19.1546666667
        },
        {
            "load_time": "2024-09-02 04:15:00",
            "load_data": 18.8286666667
        },
        {
            "load_time": "2024-09-02 04:30:00",
            "load_data": 19.246
        },
        {
            "load_time": "2024-09-02 04:45:00",
            "load_data": 18.6893333333
        },
        {
            "load_time": "2024-09-02 05:00:00",
            "load_data": 18.608
        },
        {
            "load_time": "2024-09-02 05:15:00",
            "load_data": 18.8486666667
        },
        {
            "load_time": "2024-09-02 05:30:00",
            "load_data": 18.5786666667
        },
        {
            "load_time": "2024-09-02 05:45:00",
            "load_data": 17.9953333333
        },
        {
            "load_time": "2024-09-02 06:00:00",
            "load_data": 13.6613333333
        },
        {
            "load_time": "2024-09-02 06:15:00",
            "load_data": 10.78
        },
        {
            "load_time": "2024-09-02 06:30:00",
            "load_data": 11.4146666667
        },
        {
            "load_time": "2024-09-02 06:45:00",
            "load_data": 5.834
        },
        {
            "load_time": "2024-09-02 07:00:00",
            "load_data": 58.8693333333
        },
        {
            "load_time": "2024-09-02 07:15:00",
            "load_data": 113.7306666667
        },
        {
            "load_time": "2024-09-02 07:30:00",
            "load_data": 150.1213333333
        },
        {
            "load_time": "2024-09-02 07:45:00",
            "load_data": 196.6926666667
        },
        {
            "load_time": "2024-09-02 08:00:00",
            "load_data": 216.8506666667
        }
    ]
}
        self.client.post(url, json=data)
