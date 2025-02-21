from locust import HttpUser, task, between


class VppUser(HttpUser):
    host = "http://127.0.0.1:5555"
    wait_time = between(10000,10001)

    def on_start(self):
        print("start to test")

    def on_stop(self):
        print("stopping...")

    @task
    def payload(self):
        url = "/ustlf/station/model_test"
        data = {
    "site_id": "1770070031883964416",
    "real_his_load": [
        {
    "load_time": "2024-6-19 17:45:00",
    "load_data": 298.6173333
  }
    ]
}
        self.client.post(url, json=data)
