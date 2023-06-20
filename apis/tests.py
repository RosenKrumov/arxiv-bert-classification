from django.test import SimpleTestCase
from django.urls import reverse


class PredictionTests(SimpleTestCase):
    def test_get_returns_not_allowed(self):
        response = self.client.get(reverse("prediction"))
        self.assertEqual(response.status_code, 405)

    def test_empty_input_returns_bad_request(self):
        response = self.client.post(
            reverse("prediction"),
            data={},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_full_input_returns_ok(self):
        response = self.client.post(
            reverse("prediction"),
            data={"text": "This is a test input."},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
