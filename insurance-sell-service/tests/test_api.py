from http import HTTPStatus


def test_predict_must_return_a_predict(http_client):
    response = http_client.post(
        '/predict/',
        json={
            'Gender': 'Female',
            'Age': 20,
            'HasDrivingLicense': 1,
            'Switch': 1,
            'VehicleAge': '< 1 Year',
            'PastAccident': 'Yes',
            'AnnualPremium': '£2,000',
            'cohort': 0.5,
        },
    )

    assert 'predicted' in response.json()


def test_metrics_must_return_metrics(http_client):
    response = http_client.get('/metrics/train')

    assert response.status_code == HTTPStatus.OK
