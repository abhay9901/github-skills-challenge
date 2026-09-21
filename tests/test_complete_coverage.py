import json
import runpy

import pytest

from src.aiops_pipeline import load_data, run_pipeline
from src.anomaly_detector import AnomalyDetector
from src.calculations import area_of_circle, get_nth_fibonacci
from src.event_producer import EventProducer
from src.event_topic import EventTopic


def test_load_data_and_run_pipeline(tmp_path):
    records = [
        {
            "timestamp": "2026-09-20T10:00:00",
            "service": "payment-service",
            "response_time_ms": 120,
            "cpu_percent": 42,
            "memory_percent": 51,
            "log_level": "INFO",
            "message": "Payment request processed successfully",
        },
        {
            "timestamp": "2026-09-20T10:05:00",
            "service": "payment-service",
            "response_time_ms": 610,
            "cpu_percent": 75,
            "memory_percent": 70,
            "log_level": "ERROR",
            "message": "Payment service timeout",
        },
    ]
    data_file = tmp_path / "service_data.json"
    data_file.write_text(json.dumps(records), encoding="utf-8")

    assert load_data(data_file) == records
    result = run_pipeline(data_file)

    assert result["records_processed"] == 2
    assert len(result["anomalies_detected"]) == 1
    assert result["events_consumed"] == []


def test_pipeline_script_entry_point(capsys, monkeypatch):
    class ConsumerWithEvents:
        def __init__(self, topic):
            self.topic = topic

        def consume(self):
            return [
                {
                    "service": "payment-service",
                    "timestamp": "2026-09-20T10:05:00",
                    "type": "ANOMALY",
                    "reasons": ["High response time"],
                }
            ]

    monkeypatch.setattr("event_consumer.EventConsumer", ConsumerWithEvents)
    runpy.run_path("src/aiops_pipeline.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "AIOps Pipeline Result" in output
    assert "Records processed:" in output
    assert "Service: payment-service" in output


def test_detector_covers_all_thresholds_and_warning():
    detector = AnomalyDetector()
    record = {
        "timestamp": "2026-09-20T10:05:00",
        "service": "payment-service",
        "response_time_ms": 501,
        "cpu_percent": 81,
        "memory_percent": 81,
        "log_level": "WARNING",
        "message": "Payment service warning",
    }

    event = detector.detect(record)

    assert event["reasons"] == [
        "High response time",
        "High CPU utilization",
        "High memory utilization",
        "Error log detected",
    ]


def test_producer_rejects_empty_event():
    producer = EventProducer(EventTopic("anomaly-events"))

    assert producer.publish(None) is False


def test_topic_clear_removes_messages():
    topic = EventTopic("anomaly-events")
    topic.publish({"type": "ANOMALY"})

    topic.clear()

    assert topic.get_messages() == []


def test_calculation_error_and_iterative_branches():
    with pytest.raises(ValueError):
        area_of_circle(-1)
    with pytest.raises(ValueError):
        get_nth_fibonacci(-1)

    assert get_nth_fibonacci(10) == 55
    assert area_of_circle(2) == pytest.approx(12.566370614359172)
