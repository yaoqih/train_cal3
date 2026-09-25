import hashlib
import json
import random
from pathlib import Path

from fzd_shunting.data import slot_witness
from fzd_shunting.domain import Yard
from fzd_shunting.dispatch import Dispatcher
from fzd_shunting.environment import Environment
from fzd_shunting.request import load_request, normalize_request


def test_copied_dataset_merging_and_augmented_targets():
    base = Path(__file__).resolve().parents[1] / "data" / "point_to_area"
    manifest = json.loads((base / "manifest.json").read_text())
    yard = Yard.load()
    assert len(manifest["files"]) == 96
    for entry in manifest["files"]:
        raw_path = base / "raw" / entry["file"]
        assert (
            hashlib.sha256(raw_path.read_bytes()).hexdigest() == entry["source_sha256"]
        )
        raw = json.loads(raw_path.read_text())
        normalized = json.loads((base / "normalized" / entry["file"]).read_text())
        augmented = json.loads((base / "augmented" / entry["file"]).read_text())
        # Raw files are provenance archives, not a second supported runtime schema.
        assert {str(v["No"]) for v in raw["StartStatus"]} == {
            v["No"] for v in normalized["StartStatus"]
        }
        assert normalize_request(normalized) == normalized
        for north, south, merged in [
            ("存5线北", "存5线南", "存5线"),
            ("洗罐线北", "洗罐站", "洗罐线"),
        ]:
            expected = []
            for line in (north, south):
                expected.extend(
                    str(v["No"])
                    for v in sorted(
                        (v for v in raw["StartStatus"] if v["Line"] == line),
                        key=lambda v: v["Position"],
                    )
                )
            actual = [v for v in normalized["StartStatus"] if v["Line"] == merged]
            assert [v["No"] for v in actual] == expected
            assert [v["Position"] for v in actual] == list(range(1, len(actual) + 1))
        normal_by_no = {v["No"]: v for v in normalized["StartStatus"]}
        for vehicle in augmented["StartStatus"]:
            original = normal_by_no[vehicle["No"]]
            assert set(vehicle["TargetLines"]) == set(original["TargetLines"])
            assert {k: v for k, v in vehicle.items() if k != "TargetLines"} == {
                k: v for k, v in original.items() if k != "TargetLines"
            }
        scenario = load_request(augmented, yard)
        assert len(scenario.protected) == entry["protected"]
        assert (
            Environment(yard, scenario).validate_state(scenario.initial)
            == entry["initial_state_errors"]
        )
        assert slot_witness(scenario, yard, random.Random(0)) is not None
        dispatcher = Dispatcher(Environment(yard, scenario))
        for line in dispatcher.protected_lines:
            stack = scenario.initial.stack(line)
            first = min(i for i, no in enumerate(stack) if no in dispatcher.protected)
            assert dispatcher.order_assignment(stack[first:], line) is not None
